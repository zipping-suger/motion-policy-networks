import sys
import numpy as np
import torch
import pickle
import argparse

# Required imports – adjust if your environment paths differ
try:
    from robofin.robots import FrankaRobot, SE3
    from utils import FrankaSampler
    from mpinets.model import MotionPolicyNetwork
    from mpinets.geometry import construct_mixed_point_cloud
    from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
    from geometrout.primitive import Cuboid, Cylinder, Sphere
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you are running in the proper environment with all modules.")
    sys.exit(1)

# Constants – must match those used during training / inference
NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150

# ----------------------------------------------------------------------
# Conversion functions (from your earlier code)
# ----------------------------------------------------------------------
def dicts_to_geometrout_primitives(primitives_list):
    """
    Convert a list of primitive dicts (as stored in the expert pickle)
    into a list of geometrout primitive objects.
    """
    result = []
    for prim in primitives_list:
        typ = prim["type"]
        center = prim["center"]                     # [x, y, z]
        quat = prim.get("quaternion", [1.0, 0.0, 0.0, 0.0])

        if typ == "cuboid":
            dims = prim["dims"]                      # [dx, dy, dz]
            result.append(Cuboid(center=center, dims=dims, quaternion=quat))
        elif typ == "cylinder":
            radius, height = prim["dims"]            # [radius, height]
            result.append(Cylinder(center=center, radius=radius,
                                   height=height, quaternion=quat))
        elif typ == "sphere":
            radius = prim["dims"][0] if isinstance(prim["dims"], (list, tuple)) else prim["dims"]
            result.append(Sphere(center=center, radius=radius))
        else:
            raise ValueError(f"Unknown primitive type: {typ}")
    return result


def get_tool_parameters_from_data(tool_data, is_start=True):
    """
    Extract tool parameters from the raw tool dictionary (saved in the expert pickle).
    Returns the same dictionary format expected by the sampling functions.
    """
    prefix = "start" if is_start else "target"
    num_prims = tool_data[f"{prefix}_tool_num_primitives"]
    return {
        "is_composite": num_prims > 1,
        "tool_dims": tool_data[f"{prefix}_tool_dims"],
        "tool_offsets": tool_data[f"{prefix}_tool_offset"],   # note: singular "offset"
        "tool_quats": tool_data[f"{prefix}_tool_quaternion"],
        "tool_num_primitives": num_prims,
    }


# ----------------------------------------------------------------------
# Sampling and point cloud functions (identical to save_trajectory_simple.py)
# ----------------------------------------------------------------------
def sample_robot_points(fk_sampler, config, tool_params, num_points):
    """Sample points on the robot (with tool) for a given configuration."""
    device = config.device if hasattr(config, 'device') else torch.device('cuda:0')
    dtype = config.dtype if hasattr(config, 'dtype') else torch.float32

    if tool_params["is_composite"] and tool_params["tool_num_primitives"] > 0:
        tool_dims = tool_params["tool_dims"]
        tool_offsets = tool_params["tool_offsets"]
        tool_quats = tool_params["tool_quats"]
    else:
        # Single primitive – wrap in lists
        if tool_params["tool_num_primitives"] > 0:
            tool_dims = [tool_params["tool_dims"]]
            tool_offsets = [tool_params["tool_offsets"]]
            tool_quats = [tool_params["tool_quats"]]
        else:
            tool_dims = []
            tool_offsets = []
            tool_quats = []

    tool_dims_tensor = torch.tensor(tool_dims, dtype=dtype, device=device)
    tool_offsets_tensor = torch.tensor(tool_offsets, dtype=dtype, device=device)
    tool_quats_tensor = torch.tensor(tool_quats, dtype=dtype, device=device)
    tool_num_primitives_tensor = torch.tensor(len(tool_dims), dtype=torch.long, device=device)

    return fk_sampler.sample_composite(
        config,
        tool_dims_tensor,
        tool_offsets_tensor,
        tool_quats_tensor,
        tool_num_primitives_tensor,
        num_points,
    )


def sample_target_points(fk_sampler, pose, tool_params, num_points):
    """Sample points on the tool at the target pose."""
    device = pose.device
    dtype = pose.dtype

    if tool_params["is_composite"] and tool_params["tool_num_primitives"] > 0:
        tool_dims = tool_params["tool_dims"]
        tool_offsets = tool_params["tool_offsets"]
        tool_quats = tool_params["tool_quats"]
    else:
        if tool_params["tool_num_primitives"] > 0:
            tool_dims = [tool_params["tool_dims"]]
            tool_offsets = [tool_params["tool_offsets"]]
            tool_quats = [tool_params["tool_quats"]]
        else:
            tool_dims = []
            tool_offsets = []
            tool_quats = []

    tool_dims_tensor = torch.tensor(tool_dims, dtype=dtype, device=device)
    tool_offsets_tensor = torch.tensor(tool_offsets, dtype=dtype, device=device)
    tool_quats_tensor = torch.tensor(tool_quats, dtype=dtype, device=device)
    tool_num_primitives_tensor = torch.tensor(len(tool_dims), dtype=torch.long, device=device)

    return fk_sampler.sample_composite_end_effector(
        pose,
        tool_dims_tensor,
        tool_offsets_tensor,
        tool_quats_tensor,
        tool_num_primitives_tensor,
        num_points,
    )


def make_point_cloud_from_primitives(
    q0: torch.Tensor,
    target,
    obstacles: list,
    fk_sampler,
    tool_params_start,
    tool_params_target,
) -> torch.Tensor:
    """Create the full point cloud (robot + obstacles + target) for the scene."""
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)

    robot_points = sample_robot_points(
        fk_sampler, q0, tool_params_start, NUM_ROBOT_POINTS
    )

    target_pose_tensor = (
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0)
    )
    target_points = sample_target_points(
        fk_sampler, target_pose_tensor, tool_params_target, NUM_TARGET_POINTS
    )

    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
            2 * torch.ones(NUM_TARGET_POINTS, 4),
        ),
        dim=0,
    )
    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    xyz[NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS, :3] = (
        torch.as_tensor(obstacle_points[:, :3]).float()
    )
    xyz[NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :, :3] = target_points.float()
    return xyz


def rollout_until_success(
    model,
    q0: np.ndarray,
    target,
    point_cloud: torch.Tensor,
    fk_sampler,
    tool_params_start,
) -> np.ndarray:
    """Run the policy until the target is reached (or max steps)."""
    q = torch.as_tensor(q0).unsqueeze(0).float().cuda()
    trajectory = [q]
    q_norm = normalize_franka_joints(q)

    # Target pose input for the network
    target_position = torch.as_tensor(target.matrix[:3, 3], dtype=torch.float32)
    target_rot_mat = torch.as_tensor(target.matrix[:3, :3].flatten(), dtype=torch.float32)
    target_pose_input = (
        torch.cat((target_position, target_rot_mat), dim=0)
        .float()
        .unsqueeze(0)
        .to(q.device)
    )

    def sampler(config):
        return sample_robot_points(fk_sampler, config, tool_params_start, NUM_ROBOT_POINTS)

    for step in range(MAX_ROLLOUT_LENGTH):
        q_norm = torch.clamp(
            q_norm + model(point_cloud, q_norm, target_pose_input), min=-1, max=1
        )
        qt = unnormalize_franka_joints(q_norm)
        trajectory.append(qt)

        eff_pose = FrankaRobot.fk(
            qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper"
        )
        # Stop within 1 cm and 15 degrees of the target
        if (
            np.linalg.norm(eff_pose._xyz - target._xyz) < 0.01
            and np.abs(
                np.degrees((eff_pose.so3._quat * target.so3._quat.conjugate).radians)
            )
            < 15
        ):
            break

        # Update the robot points in the point cloud for the next step
        samples = sampler(qt).type_as(point_cloud)
        point_cloud[:, : samples.shape[1], :3] = samples

    return np.asarray([t.squeeze().detach().cpu().numpy() for t in trajectory])


# ----------------------------------------------------------------------
# Main function – modified to use first configuration from expert trajectory
# ----------------------------------------------------------------------
def solve_from_expert_pkl(
    model_path: str,
    expert_pkl: str,
    output_file: str,
    device: str = "cuda:0",
):
    """Load an expert trajectory pickle, reconstruct the scene, and solve with the model.
       The initial configuration is taken from the FIRST element of the expert trajectory,
       not from the stored 'q0' field.
    """

    # 1. Load the expert pickle
    print(f"Loading expert data from {expert_pkl}")
    with open(expert_pkl, "rb") as f:
        expert_data = pickle.load(f)

    # 2. Extract scene information
    # Use the first configuration from the expert trajectory as the starting point
    if "trajectory" not in expert_data or len(expert_data["trajectory"]) == 0:
        raise ValueError("Expert pickle does not contain a valid trajectory.")
    q0_from_traj = np.array(expert_data["trajectory"][0])   # shape (7,)
    print(f"Using first configuration from expert trajectory as q0: {q0_from_traj}")

    target_matrix = np.array(expert_data["target_matrix"])
    obstacle_dicts = expert_data["obstacles"]          # list of primitive dicts
    tool_raw = expert_data["tools"]                    # dict with start_tool_* keys

    # 3. Reconstruct target as an SE3 object
    target = SE3(matrix=target_matrix)

    # 4. Convert obstacle dicts to geometrout primitives
    obstacles = dicts_to_geometrout_primitives(obstacle_dicts)

    # 5. Extract tool parameters (start and target are the same in these pickles)
    tool_params_start = get_tool_parameters_from_data(tool_raw, is_start=True)
    tool_params_target = get_tool_parameters_from_data(tool_raw, is_start=False)

    # 6. Load the model
    print(f"Loading model from {model_path}")
    model = MotionPolicyNetwork.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    # 7. Create samplers (CPU for point cloud construction, GPU for rollout)
    cpu_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_sampler = FrankaSampler(device, use_cache=True)

    # 8. Build the point cloud for the initial configuration (from trajectory)
    print("Building point cloud...")
    point_cloud = make_point_cloud_from_primitives(
        torch.as_tensor(q0_from_traj).unsqueeze(0),
        target,
        obstacles,
        cpu_sampler,
        tool_params_start,
        tool_params_target,
    )

    # 9. Run the policy to generate a trajectory
    print("Rolling out policy...")
    trajectory = rollout_until_success(
        model,
        q0_from_traj,
        target,
        point_cloud.unsqueeze(0).to(device),
        gpu_sampler,
        tool_params_start,
    )

    # 10. Save the generated trajectory (plus scene info)
    saved_data = {
        "trajectory": trajectory.tolist(),
        "obstacles": obstacle_dicts,          # keep original dicts for compatibility
        "tools": tool_raw,
        "q0": q0_from_traj.tolist(),          # overwrite with the used q0
        "target_matrix": target_matrix.tolist(),
        "source_expert_pkl": expert_pkl,
        "model_path": model_path,
    }

    with open(output_file, "wb") as f:
        pickle.dump(saved_data, f)

    print(f"Saved generated trajectory to {output_file}")
    print(f"Trajectory length: {len(trajectory)}")
    print(f"Number of obstacles: {len(obstacles)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve a problem from an expert trajectory pickle using a trained model. "
                    "The initial configuration is taken from the first element of the expert trajectory."
    )
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("expert_pkl", type=str, help="Path to expert trajectory pickle file")
    parser.add_argument("output_file", type=str, help="Output file path for generated trajectory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    args = parser.parse_args()

    solve_from_expert_pkl(
        args.model_path,
        args.expert_pkl,
        args.output_file,
        args.device,
    )