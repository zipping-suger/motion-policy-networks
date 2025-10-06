#!/usr/bin/env python3

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

import torch
from mpinets.model import MotionPolicyNetwork
from robofin.robots import FrankaRealRobot
from robofin.pointcloud.torch import FrankaSampler
import numpy as np
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets_msgs.msg import PlanningProblem
from sensor_msgs.msg import PointCloud2, PointField
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
import time
import trimesh.transformations as tra
from functools import partial
from geometrout.transform import SE3
import argparse
from typing import List, Tuple, Any
import sensor_msgs.point_cloud2 as pc2
import os

import rospy

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75  # Changed from 75 to 150 to match run_inference.py


class Planner:
    @torch.no_grad()
    def __init__(self, mdl_file: str):
        """
        Initializes and loads the model from the checkpoint

        :param mdl_file str: The path to the model checkpoint to be loaded
        """
        self.mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_file).cuda().eval()
        self.fk_sampler = FrankaSampler("cuda:0")

    @torch.no_grad()
    def target_point_cloud(self, pose: SE3) -> torch.Tensor:
        """
        Samples target points on the gripper

        :param pose SE3: pose of gripper in world frame
        :rtype torch.Tensor: A point cloud sampled from the gripper's mesh
        """
        target_points = self.fk_sampler.sample_end_effector(
            torch.as_tensor(pose.matrix).float().cuda().unsqueeze(0),
            num_points=NUM_TARGET_POINTS,
        )
        return target_points

    @torch.no_grad()
    def plan(
        self, q0: np.ndarray, target_pose: SE3, obstacle_pc: np.ndarray
    ) -> Tuple[bool, List[List[float]]]:
        """
        Creates a trajectory rollout toward the target. Will give up after MAX_ROLLOUT_LENGTH
        prediction steps

        :param q0 np.ndarray: A 7D array (dim 7,) representing the starting config
        :param target_pose SE3: A target pose in the `right_gripper` frame
        :param obstacle_pc np.ndarray: All the obstacle points fed to the network. These should
                            be constructed by filtering out outlier points and randomly
                            downsampling to be of length NUM_OBSTACLE_POINTS
        :rtype List[List[float]]: A trajectory as a list of lists (each has 7D). Formatted
                                  as a list to be more friendly to the ROS publisher
        """
        assert obstacle_pc.shape == (NUM_OBSTACLE_POINTS, 3), (
            "You must downsample obstacle PC before passing to planner. "
            "While you're at it, filter the outliers out as well"
        )

        # Make the point cloud
        q = torch.as_tensor(q0).cuda().unsqueeze(0).float()
        robot_points = self.fk_sampler.sample(q, NUM_ROBOT_POINTS)
        target_points = self.target_point_cloud(target_pose).squeeze()
        obstacle_points = torch.as_tensor(obstacle_pc).cuda()

        point_cloud = torch.cat(
            (
                torch.zeros(NUM_ROBOT_POINTS, 4),
                torch.ones(NUM_OBSTACLE_POINTS, 4),
                2 * torch.ones(NUM_TARGET_POINTS, 4),
            ),
            dim=0,
        ).cuda()
        point_cloud[:NUM_ROBOT_POINTS, :3] = robot_points.float()
        point_cloud[NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS, :3] = (
            obstacle_points.float()
        )
        point_cloud[NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :, :3] = (
            target_points.float()
        )
        point_cloud = point_cloud.unsqueeze(0)

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
            .unsqueeze(0)
            .to(q.device)
        )

        trajectory = [q]
        q_norm = normalize_franka_joints(q)
        success = False

        # Sampler function for the loop
        def sampler(config):
            return self.fk_sampler.sample(config, NUM_ROBOT_POINTS)

        for _ in range(MAX_ROLLOUT_LENGTH):
            q_norm = torch.clamp(
                q_norm + self.mdl(point_cloud, q_norm, target_pose_input), min=-1, max=1
            )
            qt = unnormalize_franka_joints(q_norm).type_as(q)
            assert isinstance(qt, torch.Tensor)
            trajectory.append(qt)

            # Use FrankaRobot.fk from run_inference.py
            eff_pose = FrankaRealRobot.fk(
                qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper"
            )

            # Success condition from run_inference.py
            if (
                np.linalg.norm(eff_pose._xyz - target_pose._xyz) < 0.01
                and np.abs(
                    np.degrees(
                        (eff_pose.so3._quat * target_pose.so3._quat.conjugate).radians
                    )
                )
                < 15
            ):
                success = True
                break

            # Update the point cloud for the next iteration
            samples = sampler(qt).type_as(point_cloud)
            point_cloud[:, : samples.shape[1], :3] = samples

        return success, [q.squeeze().cpu().numpy().tolist() for q in trajectory]


class PlanningNode:
    def __init__(self):
        """
        Initializes the subscribers, loads the data from file, and loads the model.
        """
        rospy.init_node("mpinets_planning_node")
        time.sleep(1)

        self.planner = None
        self.base_frame = "panda_link0"

        # Get the point cloud path parameter
        point_cloud_path = rospy.get_param("~point_cloud_path", "")

        # Determine mode based on whether point_cloud_path is provided and valid
        self.use_live_pointcloud = True  # Default to live mode

        if point_cloud_path and os.path.exists(point_cloud_path):
            self.use_live_pointcloud = False
            rospy.loginfo(f"Using file pointcloud mode: {point_cloud_path}")
        else:
            if point_cloud_path:
                rospy.logwarn(
                    f"Point cloud file not found: {point_cloud_path}. Switching to live mode."
                )
            else:
                rospy.loginfo(
                    "No point_cloud_path provided. Using live pointcloud mode."
                )
            rospy.loginfo("Using live pointcloud mode")

        self.planning_problem_subscriber = rospy.Subscriber(
            "/mpinets/planning_problem",
            PlanningProblem,
            self.plan_callback,
            queue_size=1,
        )
        self.full_point_cloud_publisher = rospy.Publisher(
            "/mpinets/full_point_cloud", PointCloud2, queue_size=1
        )
        self.plan_publisher = rospy.Publisher(
            "/mpinets/plan", JointTrajectory, queue_size=1
        )

        if not self.use_live_pointcloud:
            # Load from file
            rospy.loginfo("Loading data from file")
            self.load_point_cloud_data(point_cloud_path)
            rospy.loginfo("Data loaded")
        else:
            # Subscribe to PRE-PROCESSED point cloud topic instead of raw data
            self.processed_pointcloud_subscriber = rospy.Subscriber(
                "/mpinets/processed_pointcloud",
                PointCloud2,
                self.processed_pointcloud_callback,
                queue_size=1,
            )
            self.latest_pointcloud = None
            self.latest_pointcloud_colors = None
            self.pointcloud_received = False
            rospy.loginfo(
                "Waiting for pre-processed pointcloud data from /mpinets/processed_pointcloud..."
            )

            # Start a timer to publish the pointcloud for visualization
            rospy.Timer(rospy.Duration(1.0), self.publish_pointcloud_data)

        rospy.loginfo("Loading model")
        self.planner = Planner(rospy.get_param("~mdl_path"))
        rospy.loginfo("Model loaded")
        rospy.loginfo("System ready")

    def processed_pointcloud_callback(self, msg: PointCloud2):
        """
        Callback for pre-processed pointcloud messages - minimal processing required
        """
        try:
            # Extract just the points (colors are for visualization only)
            points_list = []
            colors_list = []

            # Read points efficiently - expecting pre-processed data with exactly NUM_OBSTACLE_POINTS
            for p in pc2.read_points(
                msg, field_names=("x", "y", "z", "r", "g", "b", "a"), skip_nans=True
            ):
                points_list.append([p[0], p[1], p[2]])
                colors_list.append([p[3], p[4], p[5], p[6]])

            if points_list:
                points = np.array(points_list, dtype=np.float32)
                colors = np.array(colors_list, dtype=np.float32)

                # Ensure we have the right number of points
                if len(points) == NUM_OBSTACLE_POINTS:
                    self.latest_pointcloud = points
                    self.latest_pointcloud_colors = colors
                    self.pointcloud_received = True
                    rospy.loginfo_once("Received first pre-processed point cloud")
                else:
                    rospy.logwarn_throttle(
                        10,
                        f"Pre-processed point cloud has {len(points)} points, expected {NUM_OBSTACLE_POINTS}",
                    )

        except Exception as e:
            rospy.logerr_throttle(10, f"Error reading processed point cloud: {e}")

    def publish_pointcloud_data(self, event=None):
        """
        Publishes the latest pointcloud for visualization
        """
        if (
            self.latest_pointcloud is not None
            and self.latest_pointcloud_colors is not None
        ):
            self.publish_point_cloud_data(
                self.latest_pointcloud, self.latest_pointcloud_colors
            )

    @staticmethod
    def clean_point_cloud(
        xyz: np.ndarray, rgba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized point cloud cleaning with vectorized operations
        """
        if len(xyz) == 0:
            return np.zeros((NUM_OBSTACLE_POINTS, 3), dtype=np.float32), np.zeros(
                (NUM_OBSTACLE_POINTS, 4), dtype=np.float32
            )

        # Use single vectorized condition for workspace filtering
        workspace_mask = (
            (xyz[:, 0] > 0.1)
            & (xyz[:, 0] < 1.5)
            & (xyz[:, 1] > -1.5)
            & (xyz[:, 1] < 1.5)
            & (xyz[:, 2] > -0.05)
            & (xyz[:, 2] < 1.5)
        )

        xyz_filtered = xyz[workspace_mask]
        rgba_filtered = rgba[workspace_mask]

        n_filtered = len(xyz_filtered)

        if n_filtered > NUM_OBSTACLE_POINTS:
            # Fast random sampling without replacement
            indices = np.random.choice(
                n_filtered, size=NUM_OBSTACLE_POINTS, replace=False
            )
            return xyz_filtered[indices].astype(np.float32), rgba_filtered[
                indices
            ].astype(np.float32)
        elif n_filtered > 0:
            # Efficient repetition using numpy operations
            repeat_factor = (
                NUM_OBSTACLE_POINTS + n_filtered - 1
            ) // n_filtered  # ceiling division
            xyz_repeated = np.repeat(xyz_filtered, repeat_factor, axis=0)
            rgba_repeated = np.repeat(rgba_filtered, repeat_factor, axis=0)

            # Take exactly NUM_OBSTACLE_POINTS points
            if len(xyz_repeated) > NUM_OBSTACLE_POINTS:
                indices = np.random.choice(
                    len(xyz_repeated), size=NUM_OBSTACLE_POINTS, replace=False
                )
                return xyz_repeated[indices].astype(np.float32), rgba_repeated[
                    indices
                ].astype(np.float32)
            else:
                return xyz_repeated.astype(np.float32), rgba_repeated.astype(np.float32)
        else:
            return np.zeros((NUM_OBSTACLE_POINTS, 3), dtype=np.float32), np.zeros(
                (NUM_OBSTACLE_POINTS, 4), dtype=np.float32
            )

    def load_point_cloud_data(self, path: str):
        """
        Loads scene from a point cloud file, transforms into the
        'panda_link0' frame, stores it to the class, and starts a publishing
        loop to show it

        :param path str: The path to the point cloud file
        """

        # Load the file
        observation_data = np.load(
            path,
            allow_pickle=True,
        ).item()

        # Transform it into the "world frame," i.e. `panda_link0`
        full_pc = tra.transform_points(
            observation_data["pc"], observation_data["camera_pose"]
        )

        # Remove the robot points
        no_robot_mask = (
            observation_data["label_map"]["robot"] != observation_data["pc_label"]
        )
        scene_pc = full_pc[no_robot_mask]

        # Scale the color values to be within [0-1] and add alpha channel
        scene_colors = observation_data["pc_color"][no_robot_mask] / 255.0
        scene_colors = np.concatenate(
            (scene_colors, np.ones((len(scene_colors), 1))), axis=1
        )
        assert scene_colors.shape[1] == 4

        # Clean the pointcloud
        scene_pc, scene_colors = self.clean_point_cloud(scene_pc, scene_colors)

        rospy.Timer(
            rospy.Duration(1.0),
            partial(self.publish_point_cloud_data, scene_pc, scene_colors),
        )
        self.full_scene_pc = scene_pc
        self.full_scene_colors = scene_colors

    def publish_point_cloud_data(
        self, points: np.ndarray, colors: np.ndarray, _: Any = None
    ):
        """
        Publishes the point cloud so that it can be visualized in Rviz

        :param points np.ndarray: The 3D locations of the point cloud (dimension N x 3)
        :param colors np.ndarray: The color values of each point (dimension N x 4)
        :param _ Any: This is a parameter necessary to run this within a rospy timing
                      loop and is unused.
        """
        if len(points) == 0:
            return

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        assert points.shape[1] == 3
        assert colors.shape[1] == 4
        colors[:, -1] = 0.5
        data = np.concatenate((points, colors), axis=1).astype(dtype)
        data = data.tobytes()
        fields = [
            PointField(name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate("xyzrgba")
        ]
        header = Header(frame_id="panda_link0", stamp=rospy.Time.now())
        msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 7),
            row_step=(itemsize * 7 * points.shape[0]),
            data=data,
        )
        self.full_point_cloud_publisher.publish(msg)

    def plan_callback(self, msg: PlanningProblem):
        """
        Receives the planning problem from the interaction tool and calls the planner
        Afterward, it publishes the solution, whether or not we consider it a success

        :param msg PlanningProblem: A message describing the planning problem
        """
        if self.planner is None:
            rospy.logwarn("Model is not yet loaded and planner cannot yet be called")
            return

        total_start_time = time.time()
        rospy.loginfo("=== PLANNING CALLBACK STARTED ===")

        # Time the initial message processing
        msg_processing_start = time.time()
        q0 = np.asarray(msg.q0.position)
        target = SE3(
            xyz=[
                msg.target.transform.translation.x,
                msg.target.transform.translation.y,
                msg.target.transform.translation.z,
            ],
            quaternion=[
                msg.target.transform.rotation.w,
                msg.target.transform.rotation.x,
                msg.target.transform.rotation.y,
                msg.target.transform.rotation.z,
            ],
        )
        msg_processing_time = time.time() - msg_processing_start

        # Time the point cloud preparation
        pc_preparation_start = time.time()
        if not self.use_live_pointcloud:
            # Use file-based pointcloud
            rospy.loginfo("Using file-based point cloud")
            scene_pc, scene_colors = self.clean_point_cloud(
                self.full_scene_pc, self.full_scene_colors
            )
            pc_source = "file"
        else:
            # Use pre-processed pointcloud
            rospy.loginfo("Using pre-processed point cloud")
            if not self.pointcloud_received:
                rospy.logwarn("No pre-processed pointcloud received yet, cannot plan")
                return
            scene_pc = self.latest_pointcloud
            pc_source = "pre-processed"

        pc_preparation_time = time.time() - pc_preparation_start
        rospy.loginfo(f"Point cloud preparation time: {pc_preparation_time:.3f}s")
        rospy.loginfo(f"Point cloud source: {pc_source}, points: {len(scene_pc)}")

        # Time the actual planning
        planning_start = time.time()
        rospy.loginfo("Starting planning...")
        success, plan = self.planner.plan(q0, target, scene_pc)
        planning_time = time.time() - planning_start

        rospy.loginfo(f"Planning time: {planning_time:.3f}s")
        rospy.loginfo(f"Planning succeeded: {success}")
        rospy.loginfo(f"Trajectory length: {len(plan)} points")

        # Time the trajectory message construction
        trajectory_build_start = time.time()
        joint_trajectory = JointTrajectory()
        joint_trajectory.header.stamp = rospy.Time.now()
        joint_trajectory.header.frame_id = "panda_link0"
        joint_trajectory.joint_names = msg.joint_names

        # Define velocity and acceleration limits
        VEL_MAX = 0.1  # rad/s
        ACC_MAX = 0.05  # rad/s²

        # Calculate velocities and accelerations
        velocities = []
        accelerations = []
        dt = 0.12  # time step between points

        # Calculate velocities using finite differences
        vel_calc_start = time.time()
        for i in range(len(plan)):
            if i == 0:
                # First point has zero velocity
                velocities.append([0.0] * 7)
            else:
                vel = [(plan[i][j] - plan[i - 1][j]) / dt for j in range(7)]
                # Apply velocity limits
                vel = [max(min(v, VEL_MAX), -VEL_MAX) for v in vel]
                velocities.append(vel)
        vel_calc_time = time.time() - vel_calc_start

        # Calculate accelerations using finite differences
        acc_calc_start = time.time()
        for i in range(len(plan)):
            if i == 0 or i == len(plan) - 1:
                # First and last points have zero acceleration
                accelerations.append([0.0] * 7)
            else:
                acc = [(velocities[i + 1][j] - velocities[i][j]) / dt for j in range(7)]
                # Apply acceleration limits
                acc = [max(min(a, ACC_MAX), -ACC_MAX) for a in acc]
                accelerations.append(acc)
        acc_calc_time = time.time() - acc_calc_start

        # Set final velocity and acceleration to zero
        velocities[-1] = [0.0] * 7
        accelerations[-1] = [0.0] * 7

        # Build trajectory points
        point_build_start = time.time()
        for ii, q in enumerate(plan):
            point = JointTrajectoryPoint(
                time_from_start=rospy.Duration.from_sec(dt * ii)
            )
            point.positions = q
            point.velocities = velocities[ii]
            point.accelerations = accelerations[ii]
            joint_trajectory.points.append(point)
        point_build_time = time.time() - point_build_start

        trajectory_build_time = time.time() - trajectory_build_start

        # Time the publishing
        publish_start = time.time()
        self.plan_publisher.publish(joint_trajectory)
        publish_time = time.time() - publish_start

        # Total time
        total_time = time.time() - total_start_time
        rospy.loginfo(f"=== PLANNING CALLBACK COMPLETE ===")
        rospy.loginfo(f"Total callback time: {total_time:.3f}s")
        rospy.loginfo(f"Breakdown:")
        rospy.loginfo(
            f"  - Message processing: {msg_processing_time:.3f}s ({msg_processing_time/total_time*100:.1f}%)"
        )
        rospy.loginfo(
            f"  - Point cloud prep: {pc_preparation_time:.3f}s ({pc_preparation_time/total_time*100:.1f}%)"
        )
        rospy.loginfo(
            f"  - Planning: {planning_time:.3f}s ({planning_time/total_time*100:.1f}%)"
        )
        rospy.loginfo(
            f"  - Trajectory build: {trajectory_build_time:.3f}s ({trajectory_build_time/total_time*100:.1f}%)"
        )
        rospy.loginfo(
            f"  - Publishing: {publish_time:.3f}s ({publish_time/total_time*100:.1f}%)"
        )


if __name__ == "__main__":
    PlanningNode()
    rospy.spin()
