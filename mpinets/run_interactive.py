import numpy as np
import cv2
import time
import torch
from tqdm.auto import tqdm
from pathlib import Path
from geometrout.transform import SE3, SO3
from pyquaternion import Quaternion
import argparse
import pickle
import sys

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController, Bullet
from utils import FrankaSampler

# Updated model import
from mpinets.model import MotionPolicyNetwork
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets.geometry import construct_mixed_point_cloud
from mpinets.mpinets_types import PlanningProblem, ProblemSet
from geometrout.primitive import Cuboid, Cylinder

# Import for Meshcat visualization
import meshcat
import meshcat.geometry as g
import urchin

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75
GOAL_THRESHOLD = 0.01  # 1 cm threshold for goal reaching


def create_point_cloud(robot_points, obstacle_points, target_points):
    pc = torch.zeros(
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS,
        4,  # x,y,z + segmentation mask
        device="cuda:0",
    )
    # Robot points (mask=0)
    pc[:NUM_ROBOT_POINTS, :3] = robot_points
    pc[:NUM_ROBOT_POINTS, 3] = 0

    # Obstacle points (mask=1)
    mid_start = NUM_ROBOT_POINTS
    mid_end = mid_start + NUM_OBSTACLE_POINTS
    pc[mid_start:mid_end, :3] = obstacle_points
    pc[mid_start:mid_end, 3] = 1

    # Target points (mask=2)
    mid_end = NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS
    pc[mid_end:, :3] = target_points
    pc[mid_end:, 3] = 2

    return pc.unsqueeze(0)  # Add batch dimension


def ensure_orthogonal_rotmat_polar(target_rotmat):
    target_rotmat = target_rotmat.reshape(3, 3)
    U, _, Vt = np.linalg.svd(target_rotmat)
    orthogonal_rotmat = U @ Vt

    # Ensure determinant is +1
    if np.linalg.det(orthogonal_rotmat) < 0:
        Vt[-1, :] *= -1
        orthogonal_rotmat = U @ Vt

    return orthogonal_rotmat


def move_target_with_key(target_pose, key, pos_step=0.02, rot_step=5.0):
    moved = False
    xyz = np.array(target_pose.xyz)
    so3 = target_pose.so3

    # Position changes
    if key == ord("w"):
        xyz = xyz + np.array([0, pos_step, 0])
        moved = True
    elif key == ord("s"):
        xyz = xyz + np.array([0, -pos_step, 0])
        moved = True
    elif key == ord("a"):
        xyz = xyz + np.array([-pos_step, 0, 0])
        moved = True
    elif key == ord("d"):
        xyz = xyz + np.array([pos_step, 0, 0])
        moved = True
    elif key == ord("q"):
        xyz = xyz + np.array([0, 0, pos_step])
        moved = True
    elif key == ord("e"):
        xyz = xyz + np.array([0, 0, -pos_step])
        moved = True

    # Orientation changes (in gripper's local frame)
    elif key in [ord("u"), ord("o"), ord("i"), ord("k"), ord("j"), ord("l")]:
        rot_step_rad = np.radians(rot_step)
        R = so3.matrix

        if key == ord("u"):  # Roll +
            dR = SO3.from_rpy(rot_step_rad, 0, 0).matrix
        elif key == ord("o"):  # Roll -
            dR = SO3.from_rpy(-rot_step_rad, 0, 0).matrix
        elif key == ord("i"):  # Pitch +
            dR = SO3.from_rpy(0, rot_step_rad, 0).matrix
        elif key == ord("k"):  # Pitch -
            dR = SO3.from_rpy(0, -rot_step_rad, 0).matrix
        elif key == ord("j"):  # Yaw +
            dR = SO3.from_rpy(0, 0, rot_step_rad).matrix
        elif key == ord("l"):  # Yaw -
            dR = SO3.from_rpy(0, 0, -rot_step_rad).matrix

        R_new = R @ dR
        R_new_ortho = ensure_orthogonal_rotmat_polar(R_new)
        so3 = SO3(Quaternion(matrix=R_new_ortho))
        moved = True

    if moved:
        target_pose = SE3(xyz=xyz, so3=so3)
    return moved, target_pose


def convert_to_depth(problem: PlanningProblem, cam_pose: SE3) -> np.ndarray:
    """
    Renders a point cloud from the environment using a simulated depth camera.
    """
    sim_depth = Bullet(gui=False)
    franka_depth = sim_depth.load_robot(FrankaRobot)
    franka_depth.marionette(problem.q0)
    sim_depth.load_primitives(problem.obstacles)
    obstacle_pc = sim_depth.get_pointcloud_from_camera(
        cam_pose, remove_robot=franka_depth
    )
    sim_depth.clear_all_obstacles()
    return obstacle_pc


def update_meshcat_point_cloud(viz, point_cloud):
    """
    Update Meshcat visualization with current point cloud including robot points
    """
    # Create color array: blue for robot, green for obstacles, red for target
    point_cloud_colors = np.zeros((3, point_cloud.shape[0]))

    # Robot points (blue)
    point_cloud_colors[2, :NUM_ROBOT_POINTS] = 1  # Blue channel
    point_cloud_colors[0, :NUM_ROBOT_POINTS] = 0.2  # Slight red tint

    # Obstacle points (green)
    mid_start = NUM_ROBOT_POINTS
    mid_end = mid_start + NUM_OBSTACLE_POINTS
    point_cloud_colors[1, mid_start:mid_end] = 1  # Green for obstacles

    # Target points (red)
    point_cloud_colors[0, mid_end:] = 1  # Red for target

    # Convert all points to numpy (detach first to remove gradients)
    point_cloud_positions = point_cloud[:, :3].detach().cpu().numpy().T

    viz["point_cloud"].set_object(
        meshcat.geometry.PointCloud(
            position=point_cloud_positions,
            color=point_cloud_colors,
            size=0.005,
        )
    )


def setup_robot_meshcat_visualizer(viz, urdf_path):
    """
    Set up robot visualization in Meshcat
    """
    urdf = urchin.URDF.load(urdf_path)
    # Preload the robot meshes in meshcat at a neutral position
    for idx, (mesh, transform) in enumerate(
        urdf.visual_trimesh_fk(np.zeros(8)).items()
    ):
        viz[f"robot/{idx}"].set_object(
            g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
            g.MeshLambertMaterial(color=0xEEDD22, wireframe=False),
        )
        viz[f"robot/{idx}"].set_transform(transform)
    return urdf


def get_tool_parameters(problem):
    """
    Extract tool parameters for either single primitive or composite tools
    """
    if problem.tool is None:
        # No tool
        return {
            "is_composite": False,
            "tool_dims": [0.0, 0.0, 0.0],
            "tool_offsets": [0.0, 0.0, 0.0],
            "tool_quats": [1.0, 0.0, 0.0, 0.0],
            "tool_num_primitives": 0,
        }

    primitives = problem.tool.primitives
    if len(primitives) == 1:
        # Single primitive tool
        primitive = primitives[0]
        return {
            "is_composite": False,
            "tool_dims": primitive["dims"],
            "tool_offsets": primitive["offset"],
            "tool_quats": primitive["offset_quaternion"],
            "tool_num_primitives": 1,
        }
    else:
        # Composite tool with multiple primitives
        tool_dims = []
        tool_offsets = []
        tool_quats = []

        for primitive in primitives:
            tool_dims.append(primitive["dims"])
            tool_offsets.append(primitive["offset"])
            tool_quats.append(primitive["offset_quaternion"])

        return {
            "is_composite": True,
            "tool_dims": tool_dims,
            "tool_offsets": tool_offsets,
            "tool_quats": tool_quats,
            "tool_num_primitives": len(primitives),
        }


def sample_robot_points(gpu_fk_sampler, config, tool_params, num_points):
    """Sample robot points with appropriate method based on tool type"""
    if tool_params["is_composite"] and tool_params["tool_num_primitives"] > 0:
        # Convert to tensors
        device = config.device
        dtype = config.dtype

        tool_dims_tensor = torch.tensor(
            tool_params["tool_dims"], dtype=dtype, device=device
        )
        tool_offsets_tensor = torch.tensor(
            tool_params["tool_offsets"], dtype=dtype, device=device
        )
        tool_quats_tensor = torch.tensor(
            tool_params["tool_quats"], dtype=dtype, device=device
        )
        tool_num_primitives_tensor = torch.tensor(
            tool_params["tool_num_primitives"], dtype=torch.long, device=device
        )

        return gpu_fk_sampler.sample_composite(
            config,
            tool_dims_tensor,
            tool_offsets_tensor,
            tool_quats_tensor,
            tool_num_primitives_tensor,
            num_points,
        ).squeeze(0)
    else:
        return gpu_fk_sampler.sample(
            config,
            tool_params["tool_dims"],
            tool_params["tool_offsets"],
            tool_params["tool_quats"],
            num_points,
        ).squeeze(0)


def sample_target_points(gpu_fk_sampler, pose, tool_params, num_points):
    """Sample target points with appropriate method based on tool type"""
    if tool_params["is_composite"] and tool_params["tool_num_primitives"] > 0:
        # Convert to tensors
        device = pose.device
        dtype = pose.dtype

        tool_dims_tensor = torch.tensor(
            tool_params["tool_dims"], dtype=dtype, device=device
        )
        tool_offsets_tensor = torch.tensor(
            tool_params["tool_offsets"], dtype=dtype, device=device
        )
        tool_quats_tensor = torch.tensor(
            tool_params["tool_quats"], dtype=dtype, device=device
        )
        tool_num_primitives_tensor = torch.tensor(
            tool_params["tool_num_primitives"], dtype=torch.long, device=device
        )

        return gpu_fk_sampler.sample_composite_end_effector(
            pose,
            tool_dims_tensor,
            tool_offsets_tensor,
            tool_quats_tensor,
            tool_num_primitives_tensor,
            num_points,
        ).squeeze(0)
    else:
        return gpu_fk_sampler.sample_end_effector(
            pose,
            tool_params["tool_dims"],
            tool_params["tool_offsets"],
            tool_params["tool_quats"],
            num_points,
        ).squeeze(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mdl_path", type=str, help="A checkpoint file from training MotionPolicyNetwork"
    )
    parser.add_argument(
        "problems",
        type=str,
        help="A pickle file of sample problems that follow the PlanningProblem format",
    )
    parser.add_argument(
        "environment_type",
        choices=[
            "tabletop",
            "cubby",
            "merged-cubby",
            "dresser",
            "free",
            "pillar",
            "cabinet",
            "all",
        ],
        help="The environment class to filter problems by, or 'all' for all environments",
    )
    parser.add_argument(
        "problem_type",
        choices=["task-oriented", "neutral-start", "neutral-goal", "all"],
        help="The type of planning problem to filter by, or 'all' for all problem types",
    )
    parser.add_argument(
        "--problem_idx",
        type=int,
        default=0,
        help="The index of the problem to visualize within the filtered set of problems (default: 0)",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Use a depth camera to create the obstacle point cloud instead of primitive-based sampling.",
    )
    args = parser.parse_args()

    # Initialize Meshcat visualizer
    viz = meshcat.Visualizer()
    try:
        viz.open()
    except Exception as e:
        print(f"Failed to open Meshcat visualizer: {e}")

    # Set up robot visualization in Meshcat
    urdf = setup_robot_meshcat_visualizer(viz, FrankaRobot.urdf)

    # Load MotionPolicyNetwork
    model = MotionPolicyNetwork.load_from_checkpoint(args.mdl_path).cuda()
    model.eval()

    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)

    sim = BulletController(hz=12, substeps=20, gui=True)
    franka = sim.load_robot(FrankaRobot)
    gripper = sim.load_robot(FrankaGripper, collision_free=True)

    # Set camera
    sim.set_camera_position(yaw=-90, pitch=-30, distance=2.5, target=[0.0, 0.0, 0.5])

    # HACK: The pickle file was created with a different module structure
    # This remaps the old module path to the new one so pickle can find the classes
    from mpinets import mpinets_types

    sys.modules["data_pipeline.environments.base_environment"] = mpinets_types
    sys.modules["mpinets.data_pipeline.environments.base_environment"] = mpinets_types

    # Load problems from pickle file
    with open(args.problems, "rb") as f:
        all_problems: ProblemSet = pickle.load(f)

    # Filter problems based on environment_type and problem_type
    filtered_problems = []
    env_type_arg = args.environment_type.replace("-", "_")
    problem_type_arg = args.problem_type.replace("-", "_")

    for env_type, scene_sets in all_problems.items():
        if env_type_arg != "all" and env_type != env_type_arg:
            continue
        for prob_type, problem_list in scene_sets.items():
            if problem_type_arg != "all" and prob_type != problem_type_arg:
                continue
            filtered_problems.extend(problem_list)

    if not filtered_problems:
        print(
            f"No problems found for environment type '{args.environment_type}' and problem type '{args.problem_type}'. Exiting."
        )
        exit()

    if args.problem_idx >= len(filtered_problems) or args.problem_idx < 0:
        raise IndexError(
            f"Problem index {args.problem_idx} out of range for the filtered set. There are {len(filtered_problems)} problems available. Max index is {len(filtered_problems) - 1}."
        )

    problem: PlanningProblem = filtered_problems[args.problem_idx]
    print(
        f"\n======= Visualizing problem {args.problem_idx} "
        f"(Env: {env_type_arg}, Problem Type: {problem_type_arg}) ======="
    )

    # Get tool parameters from the problem
    tool_params = get_tool_parameters(problem)

    if tool_params["tool_num_primitives"] > 0:
        if tool_params["is_composite"]:
            print(
                f"Using composite tool with {tool_params['tool_num_primitives']} primitives"
            )
        else:
            print(f"Using single primitive tool with dims: {tool_params['tool_dims']}")
    else:
        print("No tool attached")

    # Precompute obstacle points once based on the chosen method
    if args.use_depth:
        # Define a camera pose for rendering, this one is from `run_inference.py`
        # for 'tabletop'. In a real application, this would be a real sensor pose.
        # You may need to change this based on the environment type.

        # # tabletop camera pose
        # cam_pose = SE3(
        #     xyz=[1.5031788593125708, -1.817341016921562, 1.278088299149147],
        #     quaternion=[
        #         0.8687241016192855,
        #         0.4180885960330695,
        #         0.11516106409944685,
        #         0.23928704613569252,
        #     ],
        # ).inverse

        # cubby camera pose
        cam_pose = SE3(
            xyz=[0.08307640315968651, 1.986952324350807, 0.9996085854670145],
            quaternion=[
                -0.10162310189063647,
                -0.06726290364234049,
                0.5478233048853433,
                0.8276702686337273,
            ],
        ).inverse

        # # dresser camera pose
        # cam_pose = SE3(
        #     xyz=[0.08307640315968651, 1.986952324350807, 0.9996085854670145],
        #     quaternion=[
        #         -0.10162310189063647,
        #         -0.06726290364234049,
        #         0.5478233048853433,
        #         0.8276702686337273,
        #     ],
        # ).inverse

        all_obstacle_points = convert_to_depth(problem, cam_pose)

        # Sample NUM_OBSTACLE_POINTS from the full point cloud
        if len(all_obstacle_points) > NUM_OBSTACLE_POINTS:
            random_indices = np.random.choice(
                len(all_obstacle_points), size=NUM_OBSTACLE_POINTS, replace=False
            )
            obstacle_points = all_obstacle_points[random_indices, :]
        else:
            obstacle_points = all_obstacle_points

        print("Using depth camera for obstacle point cloud.")
    else:
        obstacle_points = construct_mixed_point_cloud(
            problem.obstacles, NUM_OBSTACLE_POINTS
        )
        print("Using primitive-based point cloud.")

    obstacle_points_tensor = torch.tensor(
        obstacle_points[:, :3], dtype=torch.float32, device="cuda:0"
    )

    # Load obstacles
    sim.load_primitives(problem.obstacles, color=[0.6, 0.6, 0.6, 1], visual_only=True)
    franka.marionette(problem.q0)

    # Initial target pose
    target_franka = sim.load_robot(FrankaGripper, collision_free=True)
    target_pose = problem.target
    target_franka.marionette(target_pose)

    # Initial point cloud construction and visualization
    q0_tensor = torch.tensor(
        problem.q0, dtype=torch.float32, device="cuda:0"
    ).unsqueeze(0)

    # Use appropriate sampling method based on tool type
    initial_robot_points = sample_robot_points(
        gpu_fk_sampler, q0_tensor, tool_params, NUM_ROBOT_POINTS
    )

    target_pose_tensor = torch.tensor(
        target_pose.matrix, dtype=torch.float32, device="cuda:0"
    ).unsqueeze(0)

    # Use appropriate sampling method for target points based on tool type
    target_points = sample_target_points(
        gpu_fk_sampler, target_pose_tensor, tool_params, NUM_TARGET_POINTS
    )

    # Create the full point cloud for visualization
    initial_point_cloud = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4, device="cuda:0"),
            torch.ones(NUM_OBSTACLE_POINTS, 4, device="cuda:0"),
            2 * torch.ones(NUM_TARGET_POINTS, 4, device="cuda:0"),
        ),
        dim=0,
    )
    initial_point_cloud[:NUM_ROBOT_POINTS, :3] = initial_robot_points.float()
    initial_point_cloud[
        NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS, :3
    ] = obstacle_points_tensor.float()
    initial_point_cloud[NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :, :3] = (
        target_points.float()
    )

    update_meshcat_point_cloud(viz, initial_point_cloud)
    current_point_cloud = initial_point_cloud.clone()

    print("Use WASD (XY), QE (Z) to move position.")
    print("Use U/O (roll), I/K (pitch), J/L (yaw) to rotate gripper.")
    print("Press SPACE to plan and execute. Press ESC to quit.")

    cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control", 200, 100)
    cv2.imshow("Control", np.zeros((100, 200), dtype=np.uint8))

    policy_final_config = None

    while True:
        key = cv2.waitKey(30) & 0xFF
        moved, target_pose = move_target_with_key(target_pose, key)
        if moved:
            target_franka.marionette(target_pose)
            # Update target points and point cloud visualization
            target_pose_tensor = torch.tensor(
                target_pose.matrix, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)

            # Use appropriate sampling method for target points based on tool type
            target_points = sample_target_points(
                gpu_fk_sampler, target_pose_tensor, tool_params, NUM_TARGET_POINTS
            )

            # Update the point cloud with new target points
            current_point_cloud[NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :, :3] = (
                target_points.float()
            )
            update_meshcat_point_cloud(viz, current_point_cloud)

        sim.step()
        time.sleep(0.03)

        # Update robot mesh in Meshcat
        sim_config, _ = franka.get_joint_states()
        fk_map = urdf.visual_trimesh_fk(sim_config[:8])
        for idx, (mesh, transform) in enumerate(fk_map.items()):
            viz[f"robot/{idx}"].set_transform(transform)

        if key == 27:  # ESC
            print("Exiting interactive session.")
            break
        elif key == 32:  # SPACE
            print("Planning and executing trajectory...")

            # Get start configuration
            if policy_final_config is None:
                start_config = problem.q0
            else:
                start_config = policy_final_config

            # Convert to tensor
            current_q = torch.tensor(
                start_config, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)
            q_norm = normalize_franka_joints(current_q)

            # Construct target points
            target_pose_mat = torch.tensor(
                target_pose.matrix, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)

            # Use appropriate sampling method for target points based on tool type
            target_points = sample_target_points(
                gpu_fk_sampler, target_pose_mat, tool_params, NUM_TARGET_POINTS
            )

            # Construct the target pose input for the model
            target_position = torch.as_tensor(
                target_pose.matrix[:3, 3], dtype=torch.float32
            )
            # Use rotation matrix R9 as rotation representation
            target_rot_mat = torch.as_tensor(
                target_pose.matrix[:3, :3].flatten(), dtype=torch.float32
            )
            target_pose_input = (
                torch.cat((target_position, target_rot_mat), dim=0)
                .float()
                .unsqueeze(0)
                .to(q_norm.device)
            )

            trajectory = []
            trajectory.append(start_config.copy())

            for i in range(MAX_ROLLOUT_LENGTH):
                # Sample points using appropriate method based on tool type
                robot_points = sample_robot_points(
                    gpu_fk_sampler, current_q, tool_params, NUM_ROBOT_POINTS
                )

                # Create point cloud for visualization
                xyz_vis = torch.cat(
                    (
                        torch.zeros(NUM_ROBOT_POINTS, 4, device="cuda:0"),
                        torch.ones(NUM_OBSTACLE_POINTS, 4, device="cuda:0"),
                        2 * torch.ones(NUM_TARGET_POINTS, 4, device="cuda:0"),
                    ),
                    dim=0,
                )
                xyz_vis[:NUM_ROBOT_POINTS, :3] = robot_points.float()
                xyz_vis[
                    NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS, :3
                ] = obstacle_points_tensor.float()
                xyz_vis[NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :, :3] = (
                    target_points.float()
                )

                # Update Meshcat visualization
                update_meshcat_point_cloud(viz, xyz_vis)

                # Create point cloud for model input
                xyz = create_point_cloud(
                    robot_points, obstacle_points_tensor, target_points
                )

                # Policy prediction
                delta_q = model(xyz, q_norm, target_pose_input)
                q_norm = torch.clamp(q_norm + delta_q, min=-1, max=1)
                current_q = unnormalize_franka_joints(q_norm)
                current_config = current_q.squeeze(0).detach().cpu().numpy()
                trajectory.append(current_config.copy())

                # Update robot mesh in Meshcat
                sim_config, _ = franka.get_joint_states()
                fk_map = urdf.visual_trimesh_fk(sim_config[:8])
                for idx, (mesh, transform) in enumerate(fk_map.items()):
                    viz[f"robot/{idx}"].set_transform(transform)

                # Check termination
                current_ee = FrankaRobot.fk(current_config).xyz
                distance = np.linalg.norm(
                    np.array(current_ee) - np.array(target_pose.xyz)
                )
                if distance < GOAL_THRESHOLD:
                    print(f"Reached target in {i+1} steps!")
                    break

            print(f"Generated trajectory with {len(trajectory)} steps")
            current_point_cloud = xyz_vis.clone()
            franka.marionette(trajectory[0])
            time.sleep(0.2)

            print("Executing policy trajectory...")
            for q in tqdm(trajectory):
                franka.control_position(q)
                sim.step()

                # Update robot mesh in Meshcat
                sim_config, _ = franka.get_joint_states()
                fk_map = urdf.visual_trimesh_fk(sim_config[:8])
                for idx, (mesh, transform) in enumerate(fk_map.items()):
                    viz[f"robot/{idx}"].set_transform(transform)

                time.sleep(0.08)

            # Store final configuration
            policy_final_config = trajectory[-1]
            policy_final_ee = FrankaRobot.fk(policy_final_config).xyz
            error = np.linalg.norm(
                np.array(policy_final_ee) - np.array(target_pose.xyz)
            )
            print(f"Policy final position error: {error:.4f} m")

            # Pause at final pose
            for _ in range(10):
                sim.step()
                time.sleep(0.05)
