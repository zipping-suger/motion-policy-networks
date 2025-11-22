# save_expert_trajectory.py
import sys
import numpy as np
import torch
import pickle
import argparse
import h5py
from pathlib import Path

# Import from the available modules
try:
    from robofin.robots import FrankaRobot
    from utils import FrankaSampler
    from mpinets.geometry import construct_mixed_point_cloud
    from mpinets.data_loader import PointCloudTrajectoryDataset, DatasetType
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Make sure you're running this script in the proper environment with the required modules"
    )
    exit(1)

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128


def get_expert_trajectory(dataset, idx):
    """
    Loads the expert trajectory for a given index from the HDF5 file.
    """
    with h5py.File(str(dataset._database), "r") as f:
        trajectory = f[dataset.trajectory_key][idx]
        return trajectory


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

    cuboids = []
    cylinders = []

    # Process cuboids
    for i in range(len(cuboid_centers)):
        center = cuboid_centers[i]
        dims = cuboid_dims[i]
        quat = cuboid_quats[i]
        
        # Skip if dimensions are zero (empty obstacle)
        if not np.all(np.isclose(dims, 0)):
            cuboids.append({
                "type": "cuboid",
                "center": center.tolist(),
                "dims": dims.tolist(),
                "quaternion": quat.tolist()
            })

    # Process cylinders
    for i in range(len(cylinder_centers)):
        center = cylinder_centers[i]
        radius = float(cylinder_radii[i])
        height = float(cylinder_heights[i])
        quat = cylinder_quats[i]
        
        # Skip if radius or height is zero (empty obstacle)
        if not (np.isclose(radius, 0) or np.isclose(height, 0)):
            cylinders.append({
                "type": "cylinder",
                "center": center.tolist(),
                "dims": [radius, height],
                "quaternion": quat.tolist()
            })

    return cuboids + cylinders


def extract_tool_data_from_dataset(data):
    """
    Extract tool data from dataset sample using the same format as save_trajectory.py
    """
    start_tool_dims = data.get("start_tool_dims", torch.zeros((1, 3)))
    start_tool_offset = data.get("start_tool_offset", torch.zeros((1, 3)))
    start_tool_quat = data.get("start_tool_quaternion", torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    start_tool_num_primitives = data.get("start_tool_num_primitives", torch.tensor(1))

    # Convert to lists
    tool_dims = start_tool_dims.cpu().numpy().tolist()
    tool_offsets = start_tool_offset.cpu().numpy().tolist()
    tool_quats = start_tool_quat.cpu().numpy().tolist()
    tool_num_primitives = int(start_tool_num_primitives.cpu().numpy())

    # Handle single primitive case
    if tool_num_primitives == 1 and not isinstance(tool_dims[0], list):
        tool_dims = [tool_dims]
        tool_offsets = [tool_offsets]
        tool_quats = [tool_quats]

    return {
        "is_composite": tool_num_primitives > 1,
        "tool_dims": tool_dims,
        "tool_offsets": tool_offsets,
        "tool_quats": tool_quats,
        "tool_num_primitives": tool_num_primitives,
    }


def convert_tool_data_to_save_format(tool_data):
    """
    Convert tool data to the format used in save_trajectory.py
    """
    return {
        "start_tool_dims": tool_data["tool_dims"],
        "start_tool_offset": tool_data["tool_offsets"],
        "start_tool_quaternion": tool_data["tool_quats"],
        "start_tool_num_primitives": tool_data["tool_num_primitives"],
        "target_tool_dims": tool_data["tool_dims"],  # Same tool for start and target
        "target_tool_offset": tool_data["tool_offsets"],
        "target_tool_quaternion": tool_data["tool_quats"],
        "target_tool_num_primitives": tool_data["tool_num_primitives"],
    }


def save_expert_trajectory_for_problem(
    data_path: str,
    problem_index: int,
    output_file: str,
    dataset_type: str = "val",
):
    """Save expert trajectory and scene data for a single problem."""

    # Convert dataset type
    if dataset_type.lower() == "train":
        ds_type = DatasetType.TRAIN
    elif dataset_type.lower() == "val":
        ds_type = DatasetType.VAL
    elif dataset_type.lower() == "test":
        ds_type = DatasetType.TEST
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = PointCloudTrajectoryDataset(
        Path(data_path),
        "global_solutions",
        NUM_ROBOT_POINTS,
        NUM_OBSTACLE_POINTS,
        NUM_TARGET_POINTS,
        ds_type,
        random_scale=0.0,
    )

    # Check problem index is valid
    if problem_index < 0 or problem_index >= len(dataset):
        raise IndexError(f"Problem index {problem_index} out of range. Dataset has {len(dataset)} problems.")

    # Get data sample
    data = dataset[problem_index]
    
    # Get expert trajectory
    expert_trajectory = get_expert_trajectory(dataset, problem_index)

    # Extract obstacles
    obstacle_data = primitives_from_dataset_data(data)

    # Extract tool data
    tool_data_raw = extract_tool_data_from_dataset(data)
    tool_data = convert_tool_data_to_save_format(tool_data_raw)

    # Extract initial configuration and target
    # FIXED: Use "configuration" instead of "start_configuration"
    q0 = data["configuration"].cpu().numpy()
    
    # Compute target pose from target configuration
    target_config = data["target_configuration"].cpu().numpy()
    target_pose = FrankaRobot.fk(target_config, eff_frame="right_gripper")
    target_matrix = target_pose.matrix.tolist()

    # Prepare data to save (only basic Python types)
    saved_data = {
        "trajectory": expert_trajectory.tolist(),  # Convert to list
        "obstacles": obstacle_data,  # Simple dicts
        "tools": tool_data,  # Tool information
        "q0": q0.tolist(),  # Convert to list
        "target_matrix": target_matrix,  # Simple list
        "data_path": data_path,
        "problem_index": problem_index,
        "dataset_type": dataset_type,
        "is_expert_trajectory": True,  # Flag to indicate this is an expert trajectory
    }

    # Save to file
    with open(output_file, "wb") as f:
        pickle.dump(saved_data, f)

    print(f"Saved expert trajectory to {output_file}")
    print(f"Trajectory length: {len(expert_trajectory)}")
    print(f"Number of obstacles: {len(obstacle_data)}")
    print(f"Tool data: {tool_data}")
    print(f"Initial configuration: {q0}")
    print(f"Target configuration: {target_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save expert trajectory from dataset")
    parser.add_argument("data_path", type=str, help="Path to the pretrain data directory")
    parser.add_argument("problem_index", type=int, help="Problem index in the dataset")
    parser.add_argument("output_file", type=str, help="Output file path")
    parser.add_argument("--dataset_type", type=str, default="val", 
                       choices=["train", "val", "test"],
                       help="Dataset type: train, val, or test (default: val)")

    args = parser.parse_args()

    save_expert_trajectory_for_problem(
        args.data_path,
        args.problem_index,
        args.output_file,
        args.dataset_type,
    )