# save_trajectory_simple.py
import numpy as np
import torch
import pickle
import argparse

# Import from the available modules in your original code
try:
    from robofin.robots import FrankaRobot
    from utils import FrankaSampler
    from mpinets.model import MotionPolicyNetwork
    from mpinets.geometry import construct_mixed_point_cloud
    from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
except ImportError as e:
    print(f"Import error: {e}")
    print(
        "Make sure you're running this script in the proper environment with the required modules"
    )
    exit(1)

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150


def extract_obstacle_data(obstacles):
    """Extract simple data from obstacles (centers, sizes, types, quaternions)"""
    obstacle_data = []

    for obstacle in obstacles:
        # Extract basic obstacle properties as simple Python types
        if hasattr(obstacle, "center"):
            center = [
                float(obstacle.center[0]),
                float(obstacle.center[1]),
                float(obstacle.center[2]),
            ]

            # Extract quaternion from obstacle pose
            quaternion = [1.0, 0.0, 0.0, 0.0]  # Default identity quaternion
            if hasattr(obstacle, "pose") and hasattr(obstacle.pose, "so3"):
                # Extract quaternion in wxyz format
                if hasattr(obstacle.pose.so3, "wxyz"):
                    quaternion = [float(x) for x in obstacle.pose.so3.wxyz]
                elif hasattr(obstacle.pose.so3, "xyzw"):
                    # Convert from xyzw to wxyz format if needed
                    xyzw = obstacle.pose.so3.xyzw
                    quaternion = [
                        float(xyzw[3]),
                        float(xyzw[0]),
                        float(xyzw[1]),
                        float(xyzw[2]),
                    ]

            if hasattr(obstacle, "dims"):
                # Cuboid
                if hasattr(obstacle.dims, "x"):
                    dims = [
                        float(obstacle.dims.x),
                        float(obstacle.dims.y),
                        float(obstacle.dims.z),
                    ]
                else:
                    dims = [float(d) for d in obstacle.dims]
                obstacle_type = "cuboid"
            elif hasattr(obstacle, "radius"):
                # Cylinder
                radius = float(obstacle.radius)
                height = float(obstacle.height)
                dims = [radius, height]
                obstacle_type = "cylinder"
            else:
                # Unknown type, skip
                continue

            obstacle_data.append(
                {
                    "type": obstacle_type,
                    "center": center,
                    "dims": dims,
                    "quaternion": quaternion,  # Add quaternion to saved data
                }
            )

    return obstacle_data


def make_point_cloud_from_primitives(
    q0: torch.Tensor,
    target,
    obstacles: list,
    fk_sampler,
) -> torch.Tensor:
    """Creates point cloud for the scene."""
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)
    robot_points = fk_sampler.sample(q0, NUM_ROBOT_POINTS)

    target_points = fk_sampler.sample_end_effector(
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        num_points=NUM_TARGET_POINTS,
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
    xyz[
        NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[:, :3]).float()
    xyz[
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :,
        :3,
    ] = target_points.float()
    return xyz


def rollout_until_success(
    mdl,
    q0: np.ndarray,
    target,
    point_cloud: torch.Tensor,
    fk_sampler,
) -> np.ndarray:
    """Rolls out policy until success criteria are met."""
    q = torch.as_tensor(q0).unsqueeze(0).float().cuda()
    trajectory = [q]
    q_norm = normalize_franka_joints(q)

    def sampler(config):
        return fk_sampler.sample(config, NUM_ROBOT_POINTS)

    for i in range(MAX_ROLLOUT_LENGTH):
        q_norm = torch.clamp(
            q_norm + mdl(point_cloud, q_norm), min=-1, max=1
        )
        qt = unnormalize_franka_joints(q_norm)
        trajectory.append(qt)

        eff_pose = FrankaRobot.fk(
            qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper"
        )
        # Stop when within 1cm and 15 degrees of target
        if (
            np.linalg.norm(eff_pose._xyz - target._xyz) < 0.01
            and np.abs(
                np.degrees((eff_pose.so3._quat * target.so3._quat.conjugate).radians)
            )
            < 15
        ):
            break
        samples = sampler(qt).type_as(point_cloud)
        point_cloud[:, : samples.shape[1], :3] = samples

    return np.asarray([t.squeeze().detach().cpu().numpy() for t in trajectory])


def save_trajectory_for_problem(
    model_path: str,
    problems_file: str,
    environment_type: str,
    problem_type: str,
    problem_index: int,
    output_file: str,
):
    """Save trajectory and obstacles for a single problem."""

    # Load model
    model = MotionPolicyNetwork.load_from_checkpoint(model_path).cuda()
    model.eval()

    # Load problems
    with open(problems_file, "rb") as f:
        all_problems = pickle.load(f)

    # Get specific problem
    problem = all_problems[environment_type][problem_type][problem_index]

    # Create samplers
    cpu_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_sampler = FrankaSampler("cuda:0", use_cache=True)

    # Create point cloud and rollout trajectory
    point_cloud = make_point_cloud_from_primitives(
        torch.as_tensor(problem.q0).unsqueeze(0),
        problem.target,
        problem.obstacles,
        cpu_sampler,
    )

    trajectory = rollout_until_success(
        model,
        problem.q0,
        problem.target,
        point_cloud.unsqueeze(0).cuda(),
        gpu_sampler,
    )

    # Extract simple obstacle data (no complex objects)
    obstacle_data = extract_obstacle_data(problem.obstacles)

    # Extract target as simple matrix
    target_matrix = (
        problem.target.matrix.tolist() if hasattr(problem.target, "matrix") else None
    )

    # Prepare data to save (only basic Python types)
    saved_data = {
        "trajectory": trajectory.tolist(),  # Convert to list
        "obstacles": obstacle_data,  # Simple dicts
        "q0": problem.q0.tolist(),  # Convert to list
        "target_matrix": target_matrix,  # Simple list
        "environment_type": environment_type,
        "problem_type": problem_type,
        "problem_index": problem_index,
    }

    # Save to file
    with open(output_file, "wb") as f:
        pickle.dump(saved_data, f)

    print(f"Saved trajectory and obstacles to {output_file}")
    print(f"Trajectory length: {len(trajectory)}")
    print(f"Number of obstacles: {len(obstacle_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("problems_file", type=str, help="Path to problems pickle file")
    parser.add_argument("environment_type", type=str, help="Environment type")
    parser.add_argument("problem_type", type=str, help="Problem type")
    parser.add_argument("problem_index", type=int, help="Problem index")
    parser.add_argument("output_file", type=str, help="Output file path")

    args = parser.parse_args()

    save_trajectory_for_problem(
        args.model_path,
        args.problems_file,
        args.environment_type,
        args.problem_type,
        args.problem_index,
        args.output_file,
    )
