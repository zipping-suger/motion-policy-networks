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

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController
from robofin.pointcloud.torch import FrankaSampler

# Updated model import
from mpinets.model import MotionPolicyNetwork
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets.geometry import construct_mixed_point_cloud
from mpinets.mpinets_types import PlanningProblem, ProblemSet
from geometrout.primitive import Cuboid, Cylinder

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75
GOAL_THRESHOLD = 0.01  # 1 cm threshold for goal reaching


def create_point_cloud(robot_points, obstacle_points, target_points):
    pc = torch.zeros(
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS,
        4,  # x,y,z + segmentation mask
        device="cuda:0"
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
    if key == ord('w'):
        xyz = xyz + np.array([0, pos_step, 0])
        moved = True
    elif key == ord('s'):
        xyz = xyz + np.array([0, -pos_step, 0])
        moved = True
    elif key == ord('a'):
        xyz = xyz + np.array([-pos_step, 0, 0])
        moved = True
    elif key == ord('d'):
        xyz = xyz + np.array([pos_step, 0, 0])
        moved = True
    elif key == ord('q'):
        xyz = xyz + np.array([0, 0, pos_step])
        moved = True
    elif key == ord('e'):
        xyz = xyz + np.array([0, 0, -pos_step])
        moved = True

    # Orientation changes (in gripper's local frame)
    elif key in [ord('u'), ord('o'), ord('i'), ord('k'), ord('j'), ord('l')]:
        rot_step_rad = np.radians(rot_step)
        R = so3.matrix

        if key == ord('u'):  # Roll +
            dR = SO3.from_rpy(rot_step_rad, 0, 0).matrix
        elif key == ord('o'):  # Roll -
            dR = SO3.from_rpy(-rot_step_rad, 0, 0).matrix
        elif key == ord('i'):  # Pitch +
            dR = SO3.from_rpy(0, rot_step_rad, 0).matrix
        elif key == ord('k'):  # Pitch -
            dR = SO3.from_rpy(0, -rot_step_rad, 0).matrix
        elif key == ord('j'):  # Yaw +
            dR = SO3.from_rpy(0, 0, rot_step_rad).matrix
        elif key == ord('l'):  # Yaw -
            dR = SO3.from_rpy(0, 0, -rot_step_rad).matrix

        R_new = R @ dR
        R_new_ortho = ensure_orthogonal_rotmat_polar(R_new)
        so3 = SO3(Quaternion(matrix=R_new_ortho))
        moved = True

    if moved:
        target_pose = SE3(xyz=xyz, so3=so3)
    return moved, target_pose


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
        choices=["tabletop", "cubby", "merged-cubby", "dresser", "all"],
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
    args = parser.parse_args()

    # Load MotionPolicyNetwork
    model = MotionPolicyNetwork.load_from_checkpoint(args.mdl_path).cuda()
    model.eval()

    cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)

    sim = BulletController(hz=12, substeps=20, gui=True)
    franka = sim.load_robot(FrankaRobot)
    gripper = sim.load_robot(FrankaGripper, collision_free=True)

    # Set camera
    sim.set_camera_position(
        yaw=-90, pitch=-30, distance=2.5, target=[0.0, 0.0, 0.5]
    )

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
        print(f"No problems found for environment type '{args.environment_type}' and problem type '{args.problem_type}'. Exiting.")
        exit()

    if args.problem_idx >= len(filtered_problems) or args.problem_idx < 0:
        raise IndexError(f"Problem index {args.problem_idx} out of range for the filtered set. There are {len(filtered_problems)} problems available. Max index is {len(filtered_problems) - 1}.")

    problem: PlanningProblem = filtered_problems[args.problem_idx]
    print(f"\n======= Visualizing problem {args.problem_idx} (Env: {env_type_arg}, Problem Type: {problem_type_arg}) =======")

    # Precompute obstacle points once
    obstacle_points = construct_mixed_point_cloud(problem.obstacles, NUM_OBSTACLE_POINTS)
    obstacle_points_tensor = torch.tensor(
        obstacle_points[:, :3],
        dtype=torch.float32,
        device="cuda:0"
    )

    # Load obstacles
    sim.load_primitives(problem.obstacles, color=[0.6, 0.6, 0.6, 1], visual_only=True)
    franka.marionette(problem.q0)

    # Initial target pose
    target_franka = sim.load_robot(FrankaGripper, collision_free=True)
    target_pose = problem.target
    target_franka.marionette(target_pose)

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

        sim.step()
        time.sleep(0.03)

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
                start_config,
                dtype=torch.float32,
                device="cuda:0"
            ).unsqueeze(0)
            q_norm = normalize_franka_joints(current_q)

            trajectory = []
            trajectory.append(start_config.copy())

            for i in range(MAX_ROLLOUT_LENGTH):
                # Sample points
                robot_points = gpu_fk_sampler.sample(
                    current_q,
                    NUM_ROBOT_POINTS
                ).squeeze(0)

                target_pose_mat = torch.tensor(
                    target_pose.matrix,
                    dtype=torch.float32,
                    device="cuda:0"
                ).unsqueeze(0)
                target_points = gpu_fk_sampler.sample_end_effector(
                    target_pose_mat,
                    NUM_TARGET_POINTS
                ).squeeze(0)

                # Create point cloud
                xyz = create_point_cloud(
                    robot_points,
                    obstacle_points_tensor,
                    target_points
                )

                # Policy prediction
                delta_q = model(xyz, q_norm)
                q_norm = torch.clamp(q_norm + delta_q, min=-1, max=1)
                current_q = unnormalize_franka_joints(q_norm)
                current_config = current_q.squeeze(0).detach().cpu().numpy()
                trajectory.append(current_config.copy())

                # Check termination
                current_ee = FrankaRobot.fk(current_config).xyz
                distance = np.linalg.norm(np.array(current_ee) - np.array(target_pose.xyz))
                if distance < GOAL_THRESHOLD:
                    print(f"Reached target in {i+1} steps!")
                    break

            print(f"Generated trajectory with {len(trajectory)} steps")
            franka.marionette(trajectory[0])
            time.sleep(0.2)

            print(f"Executing policy trajectory...")
            for q in tqdm(trajectory):
                franka.control_position(q)
                sim.step()
                time.sleep(0.08)

            # Store final configuration
            policy_final_config = trajectory[-1]
            policy_final_ee = FrankaRobot.fk(policy_final_config).xyz
            error = np.linalg.norm(np.array(policy_final_ee) - np.array(target_pose.xyz))
            print(f"Policy final position error: {error:.4f} m")

            # Pause at final pose
            for _ in range(10):
                sim.step()
                time.sleep(0.05)