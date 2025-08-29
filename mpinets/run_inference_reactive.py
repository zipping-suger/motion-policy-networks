# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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

import torch
from robofin.pointcloud.torch import FrankaSampler
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
MAX_ROLLOUT_LENGTH = 150


def make_point_cloud_from_problem(
    q0: torch.Tensor,
    target: SE3,
    obstacle_points: np.ndarray,
    fk_sampler: FrankaSampler,
) -> torch.Tensor:
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
) -> torch.Tensor:
    """
    Creates the pointcloud of the scene, including the target and the robot. When performing
    a rollout, the robot points will be replaced based on the model's prediction

    :param q0 torch.Tensor: The starting configuration (dimensions [1 x 7])
    :param target SE3: The target pose in the `right_gripper` frame
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :rtype torch.Tensor: The pointcloud (dimensions
                         [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4])
    """
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


def run_policy_in_sim(
    mdl: MotionPolicyNetwork,
    sim: BulletController,
    franka: FrankaRobot,
    problem: PlanningProblem,
    point_cloud: torch.Tensor,
    gpu_fk_sampler: FrankaSampler,
    viz: Optional[meshcat.Visualizer] = None,
    urdf: Optional[urchin.URDF] = None,
) -> np.ndarray:
    """
    Operates the robot in a reactive, step-by-step perception-action loop within a simulation.
    This function is the core controller and can optionally update visualizers.

    :param mdl: The loaded motion policy network.
    :param sim: The PyBullet simulation instance.
    :param franka: The robot interface in the simulation.
    :param problem: The planning problem containing q0, target, etc.
    :param point_cloud: The initial point cloud of the scene.
    :param gpu_fk_sampler: A sampler for generating robot points on the GPU.
    :param viz: The (optional) meshcat visualizer instance.
    :param urdf: The (optional) loaded URDF for FK visualization.
    :return: The executed trajectory as a numpy array.
    """
    point_cloud = point_cloud.cuda()
    franka.marionette(problem.q0)

    executed_trajectory = [np.asarray(problem.q0, dtype=np.float32)]

    target = problem.target
    target_position = torch.as_tensor(target.matrix[:3, 3], dtype=torch.float32)
    target_rot_mat = torch.as_tensor(
        target.matrix[:3, :3].flatten(), dtype=torch.float32
    )
    target_pose_input = (
        torch.cat((target_position, target_rot_mat), dim=0)
        .float()
        .unsqueeze(0)
        .to(point_cloud.device)
    )

    for _ in range(MAX_ROLLOUT_LENGTH):
        # Get the full 9-DOF state from the simulator
        full_q_np, _ = franka.get_joint_states()

        # The policy and normalization utils operate on the 7 arm joints.
        current_q_np = full_q_np[:7]
        current_q = torch.as_tensor(current_q_np).unsqueeze(0).float().cuda()

        # Log only the 7-DOF arm config for metrics
        executed_trajectory.append(current_q_np.copy())

        # The robot point cloud is generated from the 7 arm joints
        robot_points = gpu_fk_sampler.sample(current_q, NUM_ROBOT_POINTS)
        point_cloud[:, :NUM_ROBOT_POINTS, :3] = robot_points

        # Normalize the 7-DOF state for the policy
        q_norm = normalize_franka_joints(current_q)
        q_norm_next = torch.clamp(
            q_norm + mdl(point_cloud, q_norm, target_pose_input), min=-1, max=1
        )
        q_next = unnormalize_franka_joints(q_norm_next)

        # The controller is smart enough to command just the first 7 joints
        franka.control_position(q_next.squeeze().detach().cpu().numpy())
        sim.step()

        sim_config, _ = franka.get_joint_states()

        if viz and urdf:
            for idx, (k, v) in enumerate(
                urdf.visual_trimesh_fk(sim_config[:8]).items()
            ):
                viz[f"robot/{idx}"].set_transform(v)
            time.sleep(0.08)

        eff_pose = FrankaRobot.fk(sim_config, eff_frame="right_gripper")
        if (
            np.linalg.norm(eff_pose._xyz - target._xyz) < 0.01
            and np.abs(
                np.degrees((eff_pose.so3._quat * target.so3._quat.conjugate).radians)
            )
            < 15
        ):
            if viz:
                print("Success: Target reached.")
            break

    if viz:
        for _ in range(20):
            sim.step()
            sim_config, _ = franka.get_joint_states()
            if urdf:
                for idx, (k, v) in enumerate(
                    urdf.visual_trimesh_fk(sim_config[:8]).items()
                ):
                    viz[f"robot/{idx}"].set_transform(v)
            time.sleep(0.08)

    return np.asarray(executed_trajectory)


def convert_primitive_problems_to_depth(problems: ProblemSet):
    """
    Converts the planning problems in place from primitive-based to point-cloud-based.
    This used PyBullet to create the scene and sample a depth image. That depth image is
    then turned into a point cloud with ray casting.
    """
    # This function implementation remains the same
    print("Converting primitive problems to depth")
    sim = Bullet()
    franka = sim.load_robot(FrankaRobot)
    total_problems = 0
    for scene_sets in problems.values():
        for problem_set in scene_sets.values():
            total_problems += len(problem_set)
    with tqdm(total=total_problems) as pbar:
        for environment_type, scene_sets in problems.items():
            # Camera pose setup remains the same
            if "dresser" in environment_type:
                camera = SE3(
                    xyz=[0.083, 1.987, 0.999], quaternion=[-0.101, -0.067, 0.547, 0.828]
                ).inverse
            elif "cubby" in environment_type:
                camera = SE3(
                    xyz=[0.083, 1.987, 0.999], quaternion=[-0.101, -0.067, 0.547, 0.828]
                ).inverse
            elif "tabletop" in environment_type:
                camera = SE3(
                    xyz=[1.503, -1.817, 1.278], quaternion=[0.868, 0.418, 0.115, 0.239]
                ).inverse
            else:
                raise NotImplementedError(
                    f"Camera angle not implemented for: {environment_type}"
                )

            for problem_set in scene_sets.values():
                for p in problem_set:
                    franka.marionette(p.q0)
                    sim.load_primitives(p.obstacles)
                    p.obstacle_point_cloud = sim.get_pointcloud_from_camera(
                        camera, remove_robot=franka
                    )
                    sim.clear_all_obstacles()
                    pbar.update(1)


@torch.no_grad()
def calculate_metrics(mdl_path: str, problems: ProblemSet):
    """
    Calculates and prints metrics by running the policy in a headless simulation.
    """
    mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
    mdl.eval()
    cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
    eval = Evaluator()

    sim = BulletController(hz=5, substeps=120, gui=False)
    franka = sim.load_robot(FrankaRobot)

    for scene_type, scene_sets in problems.items():
        for problem_type, problem_set in scene_sets.items():
            eval.create_new_group(f"{scene_type}, {problem_type}")
            for problem in tqdm(problem_set, leave=False):
                if problem.obstacles:
                    sim.load_primitives(problem.obstacles)

                if problem.obstacle_point_cloud is None:
                    point_cloud = make_point_cloud_from_primitives(
                        torch.as_tensor(problem.q0).unsqueeze(0),
                        problem.target,
                        problem.obstacles,
                        cpu_fk_sampler,
                    )
                else:
                    point_cloud = make_point_cloud_from_problem(
                        torch.as_tensor(problem.q0).unsqueeze(0),
                        problem.target,
                        problem.obstacle_point_cloud,
                        cpu_fk_sampler,
                    )

                start_time = time.time()
                trajectory = run_policy_in_sim(
                    mdl,
                    sim,
                    franka,
                    problem,
                    point_cloud.unsqueeze(0),
                    gpu_fk_sampler,
                )

                eval.evaluate_trajectory(
                    trajectory,
                    0.08,
                    problem.target,
                    problem.obstacles,
                    problem.target_volume,
                    problem.target_negative_volumes,
                    time.time() - start_time,
                )

                sim.clear_all_obstacles()

            print(f"Metrics for {scene_type}, {problem_type}")
            eval.print_group_metrics()
    print("Overall Metrics")
    eval.print_overall_metrics()


@torch.no_grad()
def visualize_results(mdl_path: str, problems: ProblemSet):
    """
    Runs and visualizes problems in Pybullet using the reactive controller.
    """
    mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
    mdl.eval()
    cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
    sim = BulletController(hz=5, substeps=120, gui=True)

    sim.set_camera_position(yaw=-70, pitch=-30, distance=1, target=[0.0, 0.0, 0.5])
    eval = Evaluator()

    viz = meshcat.Visualizer()
    urdf = urchin.URDF.load(FrankaRobot.urdf)
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
            for problem in tqdm(problem_set, leave=False):
                eval.create_new_group(f"{scene_type}, {problem_type}")

                if problem.obstacles is not None:
                    sim.load_primitives(problem.obstacles, visual_only=True)
                gripper.marionette(problem.target)

                if problem.obstacle_point_cloud is None:
                    point_cloud = make_point_cloud_from_primitives(
                        torch.as_tensor(problem.q0).unsqueeze(0),
                        problem.target,
                        problem.obstacles,
                        cpu_fk_sampler,
                    )
                else:
                    point_cloud = make_point_cloud_from_problem(
                        torch.as_tensor(problem.q0).unsqueeze(0),
                        problem.target,
                        problem.obstacle_point_cloud,
                        cpu_fk_sampler,
                    )

                point_cloud_colors = np.zeros(
                    (3, NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS)
                )
                point_cloud_colors[1, :NUM_OBSTACLE_POINTS] = 1
                point_cloud_colors[0, NUM_OBSTACLE_POINTS:] = 1
                viz["point_cloud"].set_object(
                    meshcat.geometry.PointCloud(
                        position=point_cloud[NUM_ROBOT_POINTS:, :3].numpy().T,
                        color=point_cloud_colors,
                        size=0.005,
                    )
                )

                start_time = time.time()
                trajectory = run_policy_in_sim(
                    mdl,
                    sim,
                    franka,
                    problem,
                    point_cloud.unsqueeze(0),
                    gpu_fk_sampler,
                    viz,
                    urdf,
                )

                if problem.obstacles is not None:
                    eval.evaluate_trajectory(
                        trajectory,
                        0.08,
                        problem.target,
                        problem.obstacles,
                        problem.target_volume,
                        problem.target_negative_volumes,
                        time.time() - start_time,
                    )

                sim.clear_all_obstacles()

            print(f"Metrics for {scene_type}, {problem_type}")
            eval.print_group_metrics()
    print("Overall Metrics")
    eval.print_overall_metrics()


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
    args = parser.parse_args()
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
        calculate_metrics(args.mdl_path, problems)
    else:
        # Limit the number of problems for visualization
        if args.num_visualize is not None:
            for env in problems:
                for prob_type in problems[env]:
                    problems[env][prob_type] = problems[env][prob_type][
                        : args.num_visualize
                    ]
        time.sleep(10)
        visualize_results(args.mdl_path, problems)
