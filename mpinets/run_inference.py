import numpy as np
import time
from tqdm.auto import tqdm, trange

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import Bullet, BulletController

from pathlib import Path
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3

import pickle
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict
import argparse
import sys

import torch
from utils import FrankaSampler
from mpinets.model import MotionPolicyNetwork
from mpinets.geometry import construct_mixed_point_cloud
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets.metrics import Evaluator
from mpinets.mpinets_types import PlanningProblem, ProblemSet
import trimesh
import meshcat
import urchin

END_EFFECTOR_FRAME = "right_gripper"
NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 69


def get_tool_parameters(problem, device="cuda:0"):
    """
    Extract tool parameters for either single primitive or composite tools
    Returns tensors for compatibility with sampling functions
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
    # Convert to tensors
    device = config.device
    dtype = config.dtype

    # For both composite and single primitive, we use sample_composite
    # For single primitive, wrap parameters in lists
    if tool_params["is_composite"] and tool_params["tool_num_primitives"] > 0:
        tool_dims = tool_params["tool_dims"]
        tool_offsets = tool_params["tool_offsets"]
        tool_quats = tool_params["tool_quats"]
    else:
        # Single primitive - wrap in lists
        if tool_params["tool_num_primitives"] > 0:
            tool_dims = [tool_params["tool_dims"]]
            tool_offsets = [tool_params["tool_offsets"]]
            tool_quats = [tool_params["tool_quats"]]
        else:
            # No tool
            tool_dims = []
            tool_offsets = []
            tool_quats = []

    tool_dims_tensor = torch.tensor(tool_dims, dtype=dtype, device=device)
    tool_offsets_tensor = torch.tensor(tool_offsets, dtype=dtype, device=device)
    tool_quats_tensor = torch.tensor(tool_quats, dtype=dtype, device=device)
    tool_num_primitives_tensor = torch.tensor(
        len(tool_dims), dtype=torch.long, device=device
    )

    return gpu_fk_sampler.sample_composite(
        config,
        tool_dims_tensor,
        tool_offsets_tensor,
        tool_quats_tensor,
        tool_num_primitives_tensor,
        num_points,
    )


def sample_target_points(gpu_fk_sampler, pose, tool_params, num_points):
    """Sample target points with appropriate method based on tool type"""
    # Convert to tensors
    device = pose.device
    dtype = pose.dtype

    # For both composite and single primitive, we use sample_composite_end_effector
    # For single primitive, wrap parameters in lists
    if tool_params["is_composite"] and tool_params["tool_num_primitives"] > 0:
        tool_dims = tool_params["tool_dims"]
        tool_offsets = tool_params["tool_offsets"]
        tool_quats = tool_params["tool_quats"]
    else:
        # Single primitive - wrap in lists
        if tool_params["tool_num_primitives"] > 0:
            tool_dims = [tool_params["tool_dims"]]
            tool_offsets = [tool_params["tool_offsets"]]
            tool_quats = [tool_params["tool_quats"]]
        else:
            # No tool
            tool_dims = []
            tool_offsets = []
            tool_quats = []

    tool_dims_tensor = torch.tensor(tool_dims, dtype=dtype, device=device)
    tool_offsets_tensor = torch.tensor(tool_offsets, dtype=dtype, device=device)
    tool_quats_tensor = torch.tensor(tool_quats, dtype=dtype, device=device)
    tool_num_primitives_tensor = torch.tensor(
        len(tool_dims), dtype=torch.long, device=device
    )

    return gpu_fk_sampler.sample_composite_end_effector(
        pose,
        tool_dims_tensor,
        tool_offsets_tensor,
        tool_quats_tensor,
        tool_num_primitives_tensor,
        num_points,
    )


def make_point_cloud_from_problem(
    q0: torch.Tensor,
    target: SE3,
    obstacle_points: np.ndarray,
    fk_sampler: FrankaSampler,
    tool_params: dict,
) -> torch.Tensor:
    robot_points = sample_robot_points(
        fk_sampler, q0, tool_params, NUM_ROBOT_POINTS
    ).squeeze(0)

    target_points = sample_target_points(
        fk_sampler,
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        tool_params,
        NUM_TARGET_POINTS,
    ).squeeze(0)

    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
            2 * torch.ones(NUM_TARGET_POINTS, 4),
        ),
        dim=0,
    )
    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    random_obstacle_indices = np.random.choice(
        len(obstacle_points), size=NUM_OBSTACLE_POINTS, replace=False
    )
    xyz[
        NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[random_obstacle_indices, :3]).float()
    xyz[
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :,
        :3,
    ] = target_points.float()
    return xyz


def make_point_cloud_from_primitives(
    q0: torch.Tensor,
    target: SE3,
    obstacles: List[Union[Cuboid, Cylinder]],
    fk_sampler: FrankaSampler,
    tool_params: dict,
) -> torch.Tensor:
    """
    Creates the pointcloud of the scene, including the target and the robot. When performing
    a rollout, the robot points will be replaced based on the model's prediction

    :param q0 torch.Tensor: The starting configuration (dimensions [1 x 7])
    :param target SE3: The target pose in the `right_gripper` frame
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :param tool_params dict: Tool parameters from get_tool_parameters
    :rtype torch.Tensor: The pointcloud (dimensions
                         [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4])
    """
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)
    robot_points = sample_robot_points(
        fk_sampler, q0, tool_params, NUM_ROBOT_POINTS
    ).squeeze(0)

    target_points = sample_target_points(
        fk_sampler,
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        tool_params,
        NUM_TARGET_POINTS,
    ).squeeze(0)

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
    mdl: MotionPolicyNetwork,
    q0: np.ndarray,
    target: SE3,
    point_cloud: torch.Tensor,
    fk_sampler: FrankaSampler,
    tool_params: dict,
) -> np.ndarray:
    """
    Rolls out the policy until the success criteria are met. The criteria are that the
    end effector is within 1cm and 15 degrees of the target. Gives up after 150 prediction
    steps.

    :param mdl MotionPolicyNetwork: The policy
    :param q0 np.ndarray: The starting configuration (dimension [7])
    :param target SE3: The target in the `right_gripper` frame
    :param point_cloud torch.Tensor: The point cloud to be fed into the model. Should have
                                     dimensions [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4]
                                     and consist of the constituent points stacked in
                                     this order (robot, obstacle, target).
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :param tool_params dict: Tool parameters from get_tool_parameters
    :rtype np.ndarray: The trajectory
    """
    q = torch.as_tensor(q0).unsqueeze(0).float().cuda()
    assert q.ndim == 2
    # This block is to adapt for the case where we only want to roll out a
    # single trajectory
    trajectory = [q]
    q_norm = normalize_franka_joints(q)
    assert isinstance(q_norm, torch.Tensor)

    # Construct the target pose input for the model
    target_position = torch.as_tensor(target.matrix[:3, 3], dtype=torch.float32)
    # Use rotation matrix R9 as rotation representation
    target_rot_mat = torch.as_tensor(
        target.matrix[:3, :3].flatten(), dtype=torch.float32
    )
    target_pose_input = (
        torch.cat((target_position, target_rot_mat), dim=0)
        .float()
        .unsqueeze(0)
        .to(q.device)
    )

    success = False

    def sampler(config):
        return sample_robot_points(fk_sampler, config, tool_params, NUM_ROBOT_POINTS)

    for i in range(MAX_ROLLOUT_LENGTH):
        q_norm = torch.clamp(
            q_norm + mdl(point_cloud, q_norm, target_pose_input), min=-1, max=1
        )
        qt = unnormalize_franka_joints(q_norm)
        assert isinstance(qt, torch.Tensor)
        trajectory.append(qt)
        eff_pose = FrankaRobot.fk(
            qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper"
        )
        # Stop when the robot gets within 1cm and 15 degrees of the target
        if (
            np.linalg.norm(eff_pose._xyz - target._xyz) < 0.01
            and np.abs(
                np.degrees((eff_pose.so3._quat * target.so3._quat.conjugate).radians)
            )
            < 15
        ):
            success = True
            break
        samples = sampler(qt).type_as(point_cloud).squeeze(0)
        point_cloud[:, : samples.shape[0], :3] = samples

    final_trajectory = np.asarray([t.squeeze().detach().cpu().numpy() for t in trajectory])
    return final_trajectory, success


def convert_primitive_problems_to_depth(problems: ProblemSet):
    """
    Converts the planning problems in place from primitive-based to point-cloud-based.
    This used PyBullet to create the scene and sample a depth image. That depth image is
    then turned into a point cloud with ray casting.

    :param problems ProblemSet: The list of problems to convert
    :raises NotImplementedError: Raises an error if the environment type is not supported
    """
    print("Converting primitive problems to depth")
    sim = Bullet()
    franka = sim.load_robot(FrankaRobot)
    # These are the camera views used for evaluations in Motion Policy Networks
    # Count the problems
    total_problems = 0
    for scene_sets in problems.values():
        for problem_set in scene_sets.values():
            total_problems += len(problem_set)
    with tqdm(total=total_problems) as pbar:
        for environment_type, scene_sets in problems.items():
            if "dresser" in environment_type:
                camera = SE3(
                    xyz=[0.08307640315968651, 1.986952324350807, 0.9996085854670145],
                    quaternion=[
                        -0.10162310189063647,
                        -0.06726290364234049,
                        0.5478233048853433,
                        0.8276702686337273,
                    ],
                ).inverse
            elif "cubby" in environment_type:
                camera = SE3(
                    xyz=[0.08307640315968651, 1.986952324350807, 0.9996085854670145],
                    quaternion=[
                        -0.10162310189063647,
                        -0.06726290364234049,
                        0.5478233048853433,
                        0.8276702686337273,
                    ],
                ).inverse
            elif "tabletop" in environment_type:
                camera = SE3(
                    xyz=[1.5031788593125708, -1.817341016921562, 1.278088299149147],
                    quaternion=[
                        0.8687241016192855,
                        0.4180885960330695,
                        0.11516106409944685,
                        0.23928704613569252,
                    ],
                ).inverse
            else:
                raise NotImplementedError(
                    f"Camera angle is not implemented for environment type: {environment_type}"
                )
            for problem_set in scene_sets.values():
                for p in problem_set:
                    franka.marionette(p.q0)
                    sim.load_primitives(p.obstacles)
                    p.obstacle_point_cloud = sim.get_pointcloud_from_camera(
                        camera,
                        remove_robot=franka,
                    )
                    sim.clear_all_obstacles()
                    pbar.update(1)


def print_failure_composition(evaluator: Evaluator):
    """
    Print mutually exclusive failure composition from evaluator's stored metrics.
    Collision is checked first (accuracy not considered for collision failures).
    """
    all_success = []
    all_pos_err = []
    all_orient_err = []
    all_collision = []
    all_self_collision = []
    all_joint_limit = []
    all_physical = []
    all_skips = []

    for key, group in evaluator.groups.items():
        all_success.extend(group.get("success", []))
        all_pos_err.extend(group.get("position_error", []))
        all_orient_err.extend(group.get("orientation_error", []))
        all_collision.extend(group.get("collision", []))
        all_self_collision.extend(group.get("self_collision", []))
        all_joint_limit.extend(group.get("joint_limit_violation", []))
        all_physical.extend(group.get("physical_violations", []))
        all_skips.extend(group.get("skips", []))

    n_total = len(all_success)
    n_skips = len(all_skips)
    n_evaluated = len(all_pos_err)
    n_success = int(np.sum(all_success))

    if n_total == 0 or n_evaluated == 0:
        return

    pos_err = np.array(all_pos_err)
    orient_err = np.array(all_orient_err)
    env_coll = np.array(all_collision, dtype=bool)
    self_coll = np.array(all_self_collision, dtype=bool)
    jl_viol = np.array(all_joint_limit, dtype=bool)
    any_physical = np.array(all_physical, dtype=bool)

    pos_fail = pos_err >= 1.0
    orient_fail = orient_err >= 15.0
    no_collision = ~any_physical

    fail_env_coll = env_coll
    fail_self_coll_only = self_coll & ~env_coll
    fail_jl_only = jl_viol & ~env_coll & ~self_coll
    fail_pos_only = no_collision & pos_fail & ~orient_fail
    fail_orient_only = no_collision & ~pos_fail & orient_fail
    fail_both_acc = no_collision & pos_fail & orient_fail

    n_env_coll = int(fail_env_coll.sum())
    n_self_coll_only = int(fail_self_coll_only.sum())
    n_jl_only = int(fail_jl_only.sum())
    n_pos_only = int(fail_pos_only.sum())
    n_orient_only = int(fail_orient_only.sum())
    n_both_acc = int(fail_both_acc.sum())

    tpct = lambda n: f"{100*n/max(n_total,1):.1f}%"

    print("\n" + "=" * 80)
    print("FAILURE COMPOSITION (mutually exclusive, collision checked first)")
    print("=" * 80)
    print(f"{'Category':<52s}  {'Count':>5s}  {'% Total':>7s}")
    print("-" * 80)
    print(f"  {'Success':<50s}  {n_success:5d}  {tpct(n_success):>7s}")
    print(f"  {'Hard skip (no trajectory)':<50s}  {n_skips:5d}  {tpct(n_skips):>7s}")
    print(f"  {'Env collision (accuracy not considered)':<50s}  {n_env_coll:5d}  {tpct(n_env_coll):>7s}")
    print(f"  {'Self collision only (accuracy not considered)':<50s}  {n_self_coll_only:5d}  {tpct(n_self_coll_only):>7s}")
    print(f"  {'Joint limit violation only':<50s}  {n_jl_only:5d}  {tpct(n_jl_only):>7s}")
    print(f"  {'Position error only (>=1cm, <15deg)':<50s}  {n_pos_only:5d}  {tpct(n_pos_only):>7s}")
    print(f"  {'Orientation error only (<1cm, >=15deg)':<50s}  {n_orient_only:5d}  {tpct(n_orient_only):>7s}")
    print(f"  {'Both pos (>=1cm) & orient (>=15deg) error':<50s}  {n_both_acc:5d}  {tpct(n_both_acc):>7s}")
    print("-" * 80)
    checksum = n_success + n_skips + n_env_coll + n_self_coll_only + n_jl_only + n_pos_only + n_orient_only + n_both_acc
    print(f"  {'Checksum':<50s}  {checksum:5d}  {tpct(checksum):>7s}")
    print("=" * 80)


@torch.no_grad()
def calculate_metrics(mdl_path: str, problems: List[PlanningProblem], verbose: bool = False):
    mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
    mdl.eval()
    cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
    eval = Evaluator()

    failed_problems = []  # Track failed problems
    unsuccessful_problems = []  # Track problems that didn't reach target
    collision_problems = []  # Track problems with collisions

    for scene_type, scene_sets in problems.items():
        for problem_type, problem_set in scene_sets.items():
            eval.create_new_group(f"{scene_type}, {problem_type}")
            for problem_idx, problem in enumerate(tqdm(problem_set, leave=False)):
                try:
                    tool_params = get_tool_parameters(problem)

                    if problem.obstacle_point_cloud is None:
                        point_cloud = make_point_cloud_from_primitives(
                            torch.as_tensor(problem.q0).unsqueeze(0),
                            problem.target,
                            problem.obstacles,
                            cpu_fk_sampler,
                            tool_params,
                        )
                    else:
                        assert len(problem.obstacles) > 0
                        point_cloud = make_point_cloud_from_problem(
                            torch.as_tensor(problem.q0).unsqueeze(0),
                            problem.target,
                            problem.obstacle_point_cloud,
                            cpu_fk_sampler,
                            tool_params,
                        )
                    start_time = time.time()
                    trajectory, success = rollout_until_success(
                        mdl,
                        problem.q0,
                        problem.target,
                        point_cloud.unsqueeze(0).cuda(),
                        gpu_fk_sampler,
                        tool_params,
                    )

                    # Check collision using the evaluator
                    collision = eval.in_collision(trajectory, problem.obstacles)
                    self_collision = eval.has_self_collision(trajectory)
                    joint_limit_violation = eval.violates_joint_limits(trajectory)
                    
                    has_collision = collision or self_collision or joint_limit_violation

                    # Check if trajectory reached target
                    final_pose = FrankaRobot.fk(
                        trajectory[-1], eff_frame="right_gripper"
                    )
                    pos_error = np.linalg.norm(
                        final_pose._xyz - problem.target._xyz
                    )
                    orient_error = np.abs(
                        np.degrees(
                            (
                                final_pose.so3._quat
                                * problem.target.so3._quat.conjugate
                            ).radians
                        )
                    )

                    if not success:
                        unsuccessful_problems.append(
                            {
                                "environment": scene_type,
                                "problem_type": problem_type,
                                "index": problem_idx,
                                "position_error": pos_error,
                                "orientation_error": orient_error,
                                "trajectory_length": len(trajectory),
                                "collision": collision,
                                "self_collision": self_collision,
                                "joint_limit_violation": joint_limit_violation,
                            }
                        )
                        if verbose:
                            collision_status = []
                            if collision:
                                collision_status.append("env_collision")
                            if self_collision:
                                collision_status.append("self_collision")
                            if joint_limit_violation:
                                collision_status.append("joint_limit")
                            
                            collision_str = ", ".join(collision_status) if collision_status else "none"
                            
                            print(
                                f"UNSUCCESSFUL: Environment: {scene_type}, Type: {problem_type}, Index: {problem_idx}, "
                                f"Pos Error: {pos_error:.4f}m, Orient Error: {orient_error:.2f}deg, "
                                f"Steps: {len(trajectory)}, Collisions: {collision_str}"
                            )

                    # Track collision problems separately (even if they were successful in reaching target)
                    if has_collision:
                        collision_problems.append(
                            {
                                "environment": scene_type,
                                "problem_type": problem_type,
                                "index": problem_idx,
                                "position_error": pos_error,
                                "orientation_error": orient_error,
                                "trajectory_length": len(trajectory),
                                "collision": collision,
                                "self_collision": self_collision,
                                "joint_limit_violation": joint_limit_violation,
                                "success": success,  # Whether it reached target despite collision
                            }
                        )
                        if verbose and success:  # Print if it reached target but had collisions
                            collision_status = []
                            if collision:
                                collision_status.append("env_collision")
                            if self_collision:
                                collision_status.append("self_collision")
                            if joint_limit_violation:
                                collision_status.append("joint_limit")
                            
                            collision_str = ", ".join(collision_status)
                            print(
                                f"COLLISION BUT REACHED TARGET: Environment: {scene_type}, Type: {problem_type}, Index: {problem_idx}, "
                                f"Pos Error: {pos_error:.4f}m, Orient Error: {orient_error:.2f}deg, "
                                f"Collisions: {collision_str}"
                            )

                    eval.evaluate_trajectory(
                        trajectory,
                        0.08,  # We assume the network is to operate at roughly 12hz
                        problem.target,
                        problem.obstacles,
                        problem.target_volume,
                        problem.target_negative_volumes,
                        time.time() - start_time,
                        tool_params=tool_params  # Add this line
                    )

                except Exception as e:
                    # Log the failed problem immediately
                    error_msg = f"FAILED: Environment: {scene_type}, Type: {problem_type}, Index: {problem_idx}, Error: {str(e)}"
                    if verbose:
                        print(error_msg)
                    failed_problems.append(
                        {
                            "environment": scene_type,
                            "problem_type": problem_type,
                            "index": problem_idx,
                            "error": str(e),
                        }
                    )
                    continue

            print(f"Metrics for {scene_type}, {problem_type}")
            eval.print_group_metrics()

    # Print failed problems summary
    if failed_problems:
        print("\n" + "=" * 60)
        print("FAILED PROBLEMS (EXCEPTIONS):")
        print("=" * 60)
        for failed in failed_problems:
            print(
                f"Environment: {failed['environment']}, Type: {failed['problem_type']}, Index: {failed['index']}"
            )
            print(f"  Error: {failed['error']}")
            print("-" * 40)

    # Print unsuccessful problems summary
    if unsuccessful_problems:
        print("\n" + "=" * 60)
        print("UNSUCCESSFUL PROBLEMS (DID NOT REACH TARGET):")
        print("=" * 60)
        for unsuccessful in unsuccessful_problems:
            collision_status = []
            if unsuccessful['collision']:
                collision_status.append("env_collision")
            if unsuccessful['self_collision']:
                collision_status.append("self_collision")
            if unsuccessful['joint_limit_violation']:
                collision_status.append("joint_limit")
            
            collision_str = ", ".join(collision_status) if collision_status else "none"
            
            print(
                f"Environment: {unsuccessful['environment']}, Type: {unsuccessful['problem_type']}, Index: {unsuccessful['index']}"
            )
            print(
                f"  Position Error: {unsuccessful['position_error']:.4f}m, Orientation Error: {unsuccessful['orientation_error']:.2f}deg"
            )
            print(f"  Trajectory Length: {unsuccessful['trajectory_length']} steps")
            print(f"  Collisions: {collision_str}")
            print("-" * 40)

        # Print statistics
        total_unsuccessful = len(unsuccessful_problems)
        avg_pos_error = np.mean([p["position_error"] for p in unsuccessful_problems])
        avg_orient_error = np.mean(
            [p["orientation_error"] for p in unsuccessful_problems]
        )
        collision_count = sum(1 for p in unsuccessful_problems if p['collision'])
        self_collision_count = sum(1 for p in unsuccessful_problems if p['self_collision'])
        joint_limit_count = sum(1 for p in unsuccessful_problems if p['joint_limit_violation'])
        
        print(f"\nUnsuccessful Problems Summary:")
        print(f"  Total: {total_unsuccessful}")
        print(f"  Average Position Error: {avg_pos_error:.4f}m")
        print(f"  Average Orientation Error: {avg_orient_error:.2f}deg")
        print(f"  Environment Collisions: {collision_count}")
        print(f"  Self Collisions: {self_collision_count}")
        print(f"  Joint Limit Violations: {joint_limit_count}")

    # Print collision problems summary
    if collision_problems:
        print("\n" + "=" * 60)
        print("COLLISION PROBLEMS:")
        print("=" * 60)
        for collision_prob in collision_problems:
            collision_status = []
            if collision_prob['collision']:
                collision_status.append("env_collision")
            if collision_prob['self_collision']:
                collision_status.append("self_collision")
            if collision_prob['joint_limit_violation']:
                collision_status.append("joint_limit")
            
            collision_str = ", ".join(collision_status)
            status = "REACHED TARGET" if collision_prob['success'] else "DID NOT REACH TARGET"
            
            print(
                f"Environment: {collision_prob['environment']}, Type: {collision_prob['problem_type']}, Index: {collision_prob['index']}"
            )
            print(f"  Status: {status}")
            print(
                f"  Position Error: {collision_prob['position_error']:.4f}m, Orientation Error: {collision_prob['orientation_error']:.2f}deg"
            )
            print(f"  Collisions: {collision_str}")
            print("-" * 40)

        # Print collision statistics
        total_collisions = len(collision_problems)
        collision_reached_target = sum(1 for p in collision_problems if p['success'])
        collision_did_not_reach = total_collisions - collision_reached_target
        
        print(f"\nCollision Problems Summary:")
        print(f"  Total Collision Problems: {total_collisions}")
        print(f"  Collisions But Reached Target: {collision_reached_target}")
        print(f"  Collisions And Did Not Reach Target: {collision_did_not_reach}")

    print("Overall Metrics")
    eval.print_overall_metrics()
    print_failure_composition(eval)


@torch.no_grad()
def visualize_results(mdl_path: str, problems: ProblemSet, verbose: bool = False):
    """
    Runs a sequence of problems and visualizes the results in Pybullet

    :param mdl_path str: The path to the model
    :param problems List[PlanningProblem]: A list of problems
    """
    mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
    mdl.eval()
    cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
    sim = BulletController(hz=12, substeps=20, gui=True)

    sim.set_camera_position(yaw=-70, pitch=-30, distance=1, target=[0.0, 0.0, 0.5])
    eval = Evaluator()

    failed_problems = []  # Track failed problems
    unsuccessful_problems = []  # Track problems that didn't reach target
    collision_problems = []  # Track problems with collisions

    # Load the meshcat visualizer to visualize point cloud (Pybullet is bad at point clouds)
    viz = meshcat.Visualizer()

    # Load the FK module
    urdf = urchin.URDF.load(FrankaRobot.urdf)
    # Preload the robot meshes in meshcat at a neutral position
    for idx, (k, v) in enumerate(urdf.visual_trimesh_fk(np.zeros(8)).items()):
        viz[f"robot/{idx}"].set_object(
            meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
            meshcat.geometry.MeshLambertMaterial(color=0xEEDD22, wireframe=False),
        )
        viz[f"robot/{idx}"].set_transform(v)

    franka = sim.load_robot(FrankaRobot)
    gripper = sim.load_robot(FrankaGripper, collision_free=True)
    for scene_type, scene_sets in problems.items():
        for problem_type, problem_set in scene_sets.items():
            for problem_idx, problem in enumerate(tqdm(problem_set, leave=False)):
                try:
                    tool_params = get_tool_parameters(problem)

                    eval.create_new_group(f"{scene_type}, {problem_type}")
                    if problem.obstacle_point_cloud is None:
                        point_cloud = make_point_cloud_from_primitives(
                            torch.as_tensor(problem.q0).unsqueeze(0),
                            problem.target,
                            problem.obstacles,
                            cpu_fk_sampler,
                            tool_params,
                        )
                    else:
                        point_cloud = make_point_cloud_from_problem(
                            torch.as_tensor(problem.q0).unsqueeze(0),
                            problem.target,
                            problem.obstacle_point_cloud,
                            cpu_fk_sampler,
                            tool_params,
                        )
                    start_time = time.time()
                    trajectory, success = rollout_until_success(
                        mdl,
                        problem.q0,
                        problem.target,
                        point_cloud.unsqueeze(0).cuda(),
                        gpu_fk_sampler,
                        tool_params,
                    )

                    # Check collision using the evaluator
                    collision = eval.in_collision(trajectory, problem.obstacles)
                    self_collision = eval.has_self_collision(trajectory)
                    joint_limit_violation = eval.violates_joint_limits(trajectory)
                    
                    has_collision = collision or self_collision or joint_limit_violation

                    # Check if trajectory reached target
                    final_pose = FrankaRobot.fk(
                        trajectory[-1], eff_frame="right_gripper"
                    )
                    pos_error = np.linalg.norm(
                        final_pose._xyz - problem.target._xyz
                    )
                    orient_error = np.abs(
                        np.degrees(
                            (
                                final_pose.so3._quat
                                * problem.target.so3._quat.conjugate
                            ).radians
                        )
                    )

                    if not success:
                        unsuccessful_problems.append(
                            {
                                "environment": scene_type,
                                "problem_type": problem_type,
                                "index": problem_idx,
                                "position_error": pos_error,
                                "orientation_error": orient_error,
                                "trajectory_length": len(trajectory),
                                "collision": collision,
                                "self_collision": self_collision,
                                "joint_limit_violation": joint_limit_violation,
                            }
                        )
                        if verbose:
                            collision_status = []
                            if collision:
                                collision_status.append("env_collision")
                            if self_collision:
                                collision_status.append("self_collision")
                            if joint_limit_violation:
                                collision_status.append("joint_limit")
                            
                            collision_str = ", ".join(collision_status) if collision_status else "none"
                            
                            print(
                                f"UNSUCCESSFUL: Environment: {scene_type}, Type: {problem_type}, Index: {problem_idx}, "
                                f"Pos Error: {pos_error:.4f}m, Orient Error: {orient_error:.2f}deg, "
                                f"Steps: {len(trajectory)}, Collisions: {collision_str}"
                            )

                    # Track collision problems separately (even if they were successful in reaching target)
                    if has_collision:
                        collision_problems.append(
                            {
                                "environment": scene_type,
                                "problem_type": problem_type,
                                "index": problem_idx,
                                "position_error": pos_error,
                                "orientation_error": orient_error,
                                "trajectory_length": len(trajectory),
                                "collision": collision,
                                "self_collision": self_collision,
                                "joint_limit_violation": joint_limit_violation,
                                "success": success,  # Whether it reached target despite collision
                            }
                        )
                        if verbose and success:  # Print if it reached target but had collisions
                            collision_status = []
                            if collision:
                                collision_status.append("env_collision")
                            if self_collision:
                                collision_status.append("self_collision")
                            if joint_limit_violation:
                                collision_status.append("joint_limit")
                            
                            collision_str = ", ".join(collision_status)
                            print(
                                f"COLLISION BUT REACHED TARGET: Environment: {scene_type}, Type: {problem_type}, Index: {problem_idx}, "
                                f"Pos Error: {pos_error:.4f}m, Orient Error: {orient_error:.2f}deg, "
                                f"Collisions: {collision_str}"
                            )

                    if problem.obstacles is not None:
                        eval.evaluate_trajectory(
                            trajectory,
                            0.08,  # We assume the network is to operate at roughly 12hz
                            problem.target,
                            problem.obstacles,
                            problem.target_volume,
                            problem.target_negative_volumes,
                            time.time() - start_time,
                            tool_params=tool_params  # Add this line
                        )
                    point_cloud_colors = np.zeros(
                        (3, NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS)
                    )
                    point_cloud_colors[1, :NUM_OBSTACLE_POINTS] = 1
                    point_cloud_colors[0, NUM_OBSTACLE_POINTS:] = 1
                    viz["point_cloud"].set_object(
                        # Don't visualize robot points
                        meshcat.geometry.PointCloud(
                            position=point_cloud[NUM_ROBOT_POINTS:, :3].numpy().T,
                            color=point_cloud_colors,
                            size=0.005,
                        )
                    )
                    if problem.obstacles is not None:
                        sim.load_primitives(problem.obstacles, visual_only=True)
                    gripper.marionette(problem.target)
                    franka.marionette(trajectory[0])
                    time.sleep(0.2)
                    for q in trajectory:
                        franka.control_position(q)
                        sim.step()
                        sim_config, _ = franka.get_joint_states()
                        # Move meshes in meshcat to match PyBullet
                        for idx, (k, v) in enumerate(
                            urdf.visual_trimesh_fk(sim_config[:8]).items()
                        ):
                            viz[f"robot/{idx}"].set_transform(v)
                        time.sleep(0.08)
                    # Adding extra timesteps with no new controls to allow the simulation to
                    # converge to the final timestep's target and give the viewer time to look at
                    # it
                    for _ in range(20):
                        sim.step()
                        sim_config, _ = franka.get_joint_states()
                        # Move meshes in meshcat to match PyBullet
                        for idx, (k, v) in enumerate(
                            urdf.visual_trimesh_fk(sim_config[:8]).items()
                        ):
                            viz[f"robot/{idx}"].set_transform(v)
                        time.sleep(0.08)
                    sim.clear_all_obstacles()

                except Exception as e:
                    # Log the failed problem immediately
                    error_msg = f"FAILED: Environment: {scene_type}, Type: {problem_type}, Index: {problem_idx}, Error: {str(e)}"
                    if verbose:
                        print(error_msg)
                    failed_problems.append(
                        {
                            "environment": scene_type,
                            "problem_type": problem_type,
                            "index": problem_idx,
                            "error": str(e),
                        }
                    )
                    continue

            print(f"Metrics for {scene_type}, {problem_type}")
            eval.print_group_metrics()

    # Print failed problems summary
    if failed_problems:
        print("\n" + "=" * 60)
        print("FAILED PROBLEMS (EXCEPTIONS):")
        print("=" * 60)
        for failed in failed_problems:
            print(
                f"Environment: {failed['environment']}, Type: {failed['problem_type']}, Index: {failed['index']}"
            )
            print(f"  Error: {failed['error']}")
            print("-" * 40)

    # Print unsuccessful problems summary
    if unsuccessful_problems:
        print("\n" + "=" * 60)
        print("UNSUCCESSFUL PROBLEMS (DID NOT REACH TARGET):")
        print("=" * 60)
        for unsuccessful in unsuccessful_problems:
            collision_status = []
            if unsuccessful['collision']:
                collision_status.append("env_collision")
            if unsuccessful['self_collision']:
                collision_status.append("self_collision")
            if unsuccessful['joint_limit_violation']:
                collision_status.append("joint_limit")
            
            collision_str = ", ".join(collision_status) if collision_status else "none"
            
            print(
                f"Environment: {unsuccessful['environment']}, Type: {unsuccessful['problem_type']}, Index: {unsuccessful['index']}"
            )
            print(
                f"  Position Error: {unsuccessful['position_error']:.4f}m, Orientation Error: {unsuccessful['orientation_error']:.2f}deg"
            )
            print(f"  Trajectory Length: {unsuccessful['trajectory_length']} steps")
            print(f"  Collisions: {collision_str}")
            print("-" * 40)

        # Print statistics
        total_unsuccessful = len(unsuccessful_problems)
        avg_pos_error = np.mean([p["position_error"] for p in unsuccessful_problems])
        avg_orient_error = np.mean(
            [p["orientation_error"] for p in unsuccessful_problems]
        )
        collision_count = sum(1 for p in unsuccessful_problems if p['collision'])
        self_collision_count = sum(1 for p in unsuccessful_problems if p['self_collision'])
        joint_limit_count = sum(1 for p in unsuccessful_problems if p['joint_limit_violation'])
        
        print(f"\nUnsuccessful Problems Summary:")
        print(f"  Total: {total_unsuccessful}")
        print(f"  Average Position Error: {avg_pos_error:.4f}m")
        print(f"  Average Orientation Error: {avg_orient_error:.2f}deg")
        print(f"  Environment Collisions: {collision_count}")
        print(f"  Self Collisions: {self_collision_count}")
        print(f"  Joint Limit Violations: {joint_limit_count}")

    # Print collision problems summary
    if collision_problems:
        print("\n" + "=" * 60)
        print("COLLISION PROBLEMS:")
        print("=" * 60)
        for collision_prob in collision_problems:
            collision_status = []
            if collision_prob['collision']:
                collision_status.append("env_collision")
            if collision_prob['self_collision']:
                collision_status.append("self_collision")
            if collision_prob['joint_limit_violation']:
                collision_status.append("joint_limit")
            
            collision_str = ", ".join(collision_status)
            status = "REACHED TARGET" if collision_prob['success'] else "DID NOT REACH TARGET"
            
            print(
                f"Environment: {collision_prob['environment']}, Type: {collision_prob['problem_type']}, Index: {collision_prob['index']}"
            )
            print(f"  Status: {status}")
            print(
                f"  Position Error: {collision_prob['position_error']:.4f}m, Orientation Error: {collision_prob['orientation_error']:.2f}deg"
            )
            print(f"  Collisions: {collision_str}")
            print("-" * 40)

        # Print collision statistics
        total_collisions = len(collision_problems)
        collision_reached_target = sum(1 for p in collision_problems if p['success'])
        collision_did_not_reach = total_collisions - collision_reached_target
        
        print(f"\nCollision Problems Summary:")
        print(f"  Total Collision Problems: {total_collisions}")
        print(f"  Collisions But Reached Target: {collision_reached_target}")
        print(f"  Collisions And Did Not Reach Target: {collision_did_not_reach}")

    print("Overall Metrics")
    eval.print_overall_metrics()
    print_failure_composition(eval)


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
            "cabinet",
            "pillar",
            "all",
        ],
        help="The environment class",
    )
    parser.add_argument(
        "problem_type",
        choices=["task-oriented", "neutral-start", "neutral-goal", "all"],
        help="The type of planning problem",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help=(
            "If set, uses a partial view pointcloud rendered in Pybullet. If not set,"
            " uses pointclouds sampled from every side of the primitives in the scene"
        ),
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help=(
            "If set, will not show visuals and will only display metrics. This will be"
            " much faster because the trajectories are not displayed"
        ),
    )
    parser.add_argument(
        "--num-visualize",
        type=int,
        default=None,
        help="Number of problems to visualize (default: all)",
    )
    # Add verbose argument
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed printout for each problem",
    )
    args = parser.parse_args()

    # HACK: The pickle file was created with a different module structure
    # This remaps the old module path to the new one so pickle can find the classes
    from mpinets import mpinets_types

    sys.modules["data_pipeline.environments.base_environment"] = mpinets_types
    sys.modules["mpinets.data_pipeline.environments.base_environment"] = mpinets_types

    with open(args.problems, "rb") as f:
        problems = pickle.load(f)
    env_type = args.environment_type.replace("-", "_")
    problem_type = args.problem_type.replace("-", "_")
    if env_type != "all":
        problems = {env_type: problems[env_type]}
    if problem_type != "all":
        for k in problems.keys():
            problems[k] = {problem_type: problems[k][problem_type]}
    if args.use_depth:
        convert_primitive_problems_to_depth(problems)
    if args.skip_visuals:
        calculate_metrics(args.mdl_path, problems, verbose=args.verbose)
    else:
        # Limit the number of problems for visualization
        if args.num_visualize is not None:
            for env in problems:
                for prob_type in problems[env]:
                    problems[env][prob_type] = problems[env][prob_type][
                        : args.num_visualize
                    ]
        time.sleep(10)
        visualize_results(args.mdl_path, problems, verbose=args.verbose)