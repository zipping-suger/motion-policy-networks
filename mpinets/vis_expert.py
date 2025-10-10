import numpy as np
import time
import torch
import h5py
from tqdm.auto import tqdm
import pickle
import meshcat
import meshcat.geometry as g
import urchin
from pathlib import Path

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController
from utils import FrankaSampler
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets.geometry import construct_mixed_point_cloud
from mpinets.mpinets_types import PlanningProblem, ProblemSet
from mpinets.data_loader import PointCloudTrajectoryDataset, DatasetType

# --- Constants from run_interactive/run_inference ---
NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
# --- End Constants ---


def get_expert_trajectory(dataset, idx):
    """
    Loads the expert trajectory for a given index from the HDF5 file.
    """
    with h5py.File(str(dataset._database), "r") as f:
        # Assuming the structure is similar to run_inference.py's dataset
        trajectory = f[dataset.trajectory_key][idx]
        return trajectory


def update_meshcat_point_cloud(
    viz,
    robot_points,
    obstacle_points,
    target_points,
):
    """
    Update Meshcat visualization with current point cloud including robot, obstacles, and target.

    This function has been modified from run_interactive.py to accept separate components
    and reconstruct the full point cloud for visualization, including the robot's point cloud.
    """

    # 1. Combine points
    # Move points to CPU and convert to numpy
    if torch.is_tensor(robot_points):
        robot_pts_np = robot_points[:, :3].detach().cpu().numpy()
    else:
        robot_pts_np = robot_points[:, :3]

    if torch.is_tensor(target_points):
        target_pts_np = target_points[:, :3].detach().cpu().numpy()
    else:
        target_pts_np = target_points[:, :3]

    if torch.is_tensor(obstacle_points):
        obstacle_pts_np = obstacle_points[:, :3].detach().cpu().numpy()
    else:
        obstacle_pts_np = obstacle_points[:, :3]

    combined_points = np.vstack([robot_pts_np, obstacle_pts_np, target_pts_np])

    # 2. Create color array: blue for robot, green for obstacles, red for target
    point_cloud_colors = np.zeros((3, combined_points.shape[0]))

    # Robot points (blue) - indices 0 to NUM_ROBOT_POINTS - 1
    point_cloud_colors[2, :NUM_ROBOT_POINTS] = 1  # Blue channel
    point_cloud_colors[0, :NUM_ROBOT_POINTS] = 0.2  # Slight red tint

    # Obstacle points (green) - indices NUM_ROBOT_POINTS to mid_end - 1
    mid_start = NUM_ROBOT_POINTS
    mid_end = mid_start + NUM_OBSTACLE_POINTS
    point_cloud_colors[1, mid_start:mid_end] = 1  # Green for obstacles

    # Target points (red) - indices mid_end to end
    point_cloud_colors[0, mid_end:] = 1  # Red for target

    # 3. Update Meshcat object
    viz["point_cloud"].set_object(
        meshcat.geometry.PointCloud(
            position=combined_points.T,  # Meshcat expects (3, N)
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
    # The `urdf.visual_trimesh_fk` function is assumed to return (mesh, transform) for links
    for idx, (mesh, transform) in enumerate(
        urdf.visual_trimesh_fk(np.zeros(8)).items()
    ):
        viz[f"robot/{idx}"].set_object(
            g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
            g.MeshLambertMaterial(color=0xEEDD22, wireframe=False),
        )
        viz[f"robot/{idx}"].set_transform(transform)
    return urdf


def primitives_from_dataset_data(data):
    """
    Extracts geometric primitives (obstacles) from the dataset sample data.
    """
    cuboid_centers = data["cuboid_centers"].cpu().numpy()
    cuboid_dims = data["cuboid_dims"].cpu().numpy()
    cuboid_quats = data["cuboid_quats"].cpu().numpy()

    cylinder_centers = data["cylinder_centers"].cpu().numpy()
    cylinder_radii = data["cylinder_radii"].cpu().numpy()
    cylinder_heights = data["cylinder_heights"].cpu().numpy()
    cylinder_quats = data["cylinder_quats"].cpu().numpy()

    cuboids = [
        Cuboid(c, d, q)
        for c, d, q in zip(list(cuboid_centers), list(cuboid_dims), list(cuboid_quats))
        if not np.all(np.isclose(d, 0))
    ]

    cylinders = [
        Cylinder(c, r, h, q)
        for c, r, h, q in zip(
            list(cylinder_centers),
            list(cylinder_radii.squeeze(1)),
            list(cylinder_heights.squeeze(1)),
            list(cylinder_quats),
        )
        if not np.isclose(r, 0) and not np.isclose(h, 0)
    ]

    return cuboids + cylinders


def run_visualization_for_problem_idx(
    problem_idx, viz, urdf, dataset, sim, franka, target_franka
):
    """
    Loads, sets up, and visualizes a single planning problem.
    """
    # --- Clear Previous State from Simulation ---
    sim.clear_all_obstacles()

    # --- Load Data for Visualization ---
    data = dataset[problem_idx]
    expert_trajectory = get_expert_trajectory(dataset, problem_idx)

    # --- Construct Attached Tool from Data ---
    attached_primitive = None
    print(f"Data keys: {list(data.keys())}")
    # Check if tool information is in the dataset
    if "start_tool_dims" in data:
        tool_dims = data["start_tool_dims"].cpu().numpy()
        # Ensure the tool has volume before creating the primitive
        if not np.all(np.isclose(tool_dims, 0)):
            print("Found attached tool in dataset. Constructing primitive...")
            attached_primitive = {
                "type": "cuboid",
                "dims": tool_dims.tolist(),
                "num_points": 300,
                "offset": data["start_tool_offset"].cpu().numpy().tolist(),
                "offset_quaternion": data["start_tool_quaternion"]
                .cpu()
                .numpy()
                .tolist(),
            }
        else:
            print("Tool dimensions are zero, no tool will be attached.")
    else:
        print("No tool information found in the dataset.")

    # Setup samplers
    # The GPU sampler is needed to include the attached primitive point cloud
    try:
        gpu_fk_sampler = FrankaSampler(
            "cuda:0", use_cache=True, attached_primitive=attached_primitive
        )
        device = "cuda:0"
    except:
        print("CUDA not available, falling back to CPU")
        gpu_fk_sampler = FrankaSampler(
            "cpu", use_cache=True, attached_primitive=attached_primitive
        )
        device = "cpu"

    # Create obstacles and target pose
    obstacles = primitives_from_dataset_data(data)
    target_pose = FrankaRobot.fk(
        data["target_configuration"].cpu().numpy(), eff_frame="right_gripper"
    )

    print(f"\n======= Visualizing Expert Trajectory for problem {problem_idx} =======")
    print(f"Trajectory length: {len(expert_trajectory)} steps")

    # --- Precompute Static Point Clouds ---
    # 1. Obstacle points (static)
    obstacle_points_np = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)
    # 2. Target points (static, based on target_pose)
    target_pose_mat = torch.tensor(
        target_pose.matrix, dtype=torch.float32, device=device
    ).unsqueeze(0)
    target_points = gpu_fk_sampler.sample_end_effector(
        target_pose_mat, NUM_TARGET_POINTS
    ).squeeze(0)

    # --- Simulation Setup ---
    sim.load_primitives(obstacles, color=[0.6, 0.6, 0.6, 1], visual_only=True)
    target_franka.marionette(target_pose)
    start_config = expert_trajectory[0]
    franka.marionette(start_config)
    time.sleep(0.5)

    # --- Initial Meshcat Point Cloud Visualization ---
    initial_q_tensor = torch.tensor(
        start_config, dtype=torch.float32, device=device
    ).unsqueeze(0)
    initial_robot_points = gpu_fk_sampler.sample(
        initial_q_tensor, NUM_ROBOT_POINTS
    ).squeeze(0)
    sim_config, _ = franka.get_joint_states()
    for idx, (mesh, transform) in enumerate(
        urdf.visual_trimesh_fk(sim_config[:8]).items()
    ):
        viz[f"robot/{idx}"].set_transform(transform)
    update_meshcat_point_cloud(
        viz, initial_robot_points, obstacle_points_np, target_points
    )
    time.sleep(2.0)

    # --- Execute and Visualize Trajectory ---
    print("Executing expert trajectory...")
    for q in tqdm(expert_trajectory, leave=False):
        franka.marionette(q)
        sim.step()

        sim_config, _ = franka.get_joint_states()
        for idx, (mesh, transform) in enumerate(
            urdf.visual_trimesh_fk(sim_config[:8]).items()
        ):
            viz[f"robot/{idx}"].set_transform(transform)

        current_q_tensor = torch.tensor(
            sim_config[:7], dtype=torch.float32, device=device
        ).unsqueeze(0)
        current_robot_points = gpu_fk_sampler.sample(
            current_q_tensor, NUM_ROBOT_POINTS
        ).squeeze(0)
        update_meshcat_point_cloud(
            viz, current_robot_points, obstacle_points_np, target_points
        )
        time.sleep(0.05)

    # --- Final Pose ---
    final_q = expert_trajectory[-1]
    final_ee = FrankaRobot.fk(final_q).xyz
    error = np.linalg.norm(np.array(final_ee) - np.array(target_pose.xyz))
    print(f"Trajectory final position error: {error:.4f} m")

    for _ in range(20):
        sim.step()
        time.sleep(0.05)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize an expert trajectory and point clouds."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pretrain data directory (e.g., './pretrain_data/ompl_cubby_6k')",
    )
    parser.add_argument(
        "--problem_idx",
        type=int,
        default=None,
        help="The index of the problem to visualize. If not provided, loops over all problems.",
    )
    args = parser.parse_args()

    val_data_path = Path(args.data_path)

    # Initialize Meshcat visualizer
    viz = meshcat.Visualizer()
    try:
        viz.open()
    except Exception as e:
        print(f"Failed to open Meshcat visualizer: {e}")
        print("Meshcat might be available at http://localhost:7000/static/")

    # Set up robot visualization in Meshcat
    urdf = setup_robot_meshcat_visualizer(viz, FrankaRobot.urdf)

    # Load validation data
    print(f"Loading dataset from {val_data_path}...")
    dataset = PointCloudTrajectoryDataset(
        val_data_path,
        "global_solutions",
        NUM_ROBOT_POINTS,
        NUM_OBSTACLE_POINTS,
        NUM_TARGET_POINTS,
        DatasetType.VAL,
        random_scale=0.0,
    )

    # Setup simulation environment once
    sim = BulletController(hz=12, substeps=20, gui=True)
    franka = sim.load_robot(FrankaRobot)
    target_franka = sim.load_robot(FrankaGripper, collision_free=True)
    sim.set_camera_position(yaw=-90, pitch=-30, distance=2.5, target=[0.0, 0.0, 0.5])

    # Determine which problem indices to run
    if args.problem_idx is not None:
        if args.problem_idx >= len(dataset) or args.problem_idx < 0:
            raise IndexError(
                f"Problem index {args.problem_idx} out of range. Max index is {len(dataset) - 1}."
            )
        problem_indices = [args.problem_idx]
    else:
        print(
            f"No problem index provided. Looping over all {len(dataset)} validation problems."
        )
        problem_indices = range(len(dataset))

    # Main loop to visualize problems
    for problem_idx in problem_indices:
        try:
            run_visualization_for_problem_idx(
                problem_idx, viz, urdf, dataset, sim, franka, target_franka
            )
            if len(problem_indices) > 1:
                print(f"--- Finished problem {problem_idx}. Pausing before next. ---")
                time.sleep(2)  # Pause between problems
        except Exception as e:
            print(f"Error visualizing problem {problem_idx}: {e}")
            if len(problem_indices) > 1:
                print("Continuing to next problem...")
                time.sleep(2)
                continue
            else:
                raise e

    print("Visualization complete.")
