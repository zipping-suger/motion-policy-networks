#!/usr/bin/env python3

import torch
from mpinets.model import MotionPolicyNetwork
from robofin.robots import FrankaRealRobot
from robofin.pointcloud.torch import FrankaSampler
import numpy as np
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets_msgs.msg import PlanningProblem
from sensor_msgs.msg import PointCloud2, PointField, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header, Bool
import time
import trimesh.transformations as tra
from functools import partial
from geometrout.transform import SE3
import sensor_msgs.point_cloud2 as pc2
import os
import threading

import rospy

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128

# Global parameter: Set to True for self-feedback (simulation), False for real robot feedback
USE_SELF_FEEDBACK = False

# The neutral configuration
NEUTRAL_CONFIG = np.array(
    [
        -0.01779206,
        -0.76012354,
        0.01978261,
        -2.34205014,
        0.02984053,
        1.54119353,
        0.75344866,
        0.025,
        0.025,
    ]
)


class ReactiveController:
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
    def get_next_action(
        self, q_current: np.ndarray, target_pose: SE3, obstacle_pc: np.ndarray
    ) -> np.ndarray:
        """
        Gets the next action (joint configuration) given current state and target

        :param q_current np.ndarray: Current 7D joint configuration
        :param target_pose SE3: Target pose in the `right_gripper` frame
        :param obstacle_pc np.ndarray: Obstacle points (NUM_OBSTACLE_POINTS, 3)
        :rtype np.ndarray: Next joint configuration (7D)
        """
        assert obstacle_pc.shape == (NUM_OBSTACLE_POINTS, 3), (
            "You must downsample obstacle PC before passing to controller. "
            "While you're at it, filter the outliers out as well"
        )

        # Convert to torch tensors
        q = torch.as_tensor(q_current).cuda().unsqueeze(0).float()

        # Sample robot points
        robot_points = self.fk_sampler.sample(q, NUM_ROBOT_POINTS)

        # Get target points
        target_points = self.target_point_cloud(target_pose).squeeze()

        # Convert obstacle points
        obstacle_points = torch.as_tensor(obstacle_pc).cuda()

        # Create point cloud
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
        target_rot_mat = torch.as_tensor(
            target_pose.matrix[:3, :3].flatten(), dtype=torch.float32
        )
        target_pose_input = (
            torch.cat((target_position, target_rot_mat), dim=0)
            .unsqueeze(0)
            .to(q.device)
        )

        # Normalize current joint configuration
        q_norm = normalize_franka_joints(q)

        # Get action from model
        action = self.mdl(point_cloud, q_norm, target_pose_input)

        # Apply action and clamp
        q_next_norm = torch.clamp(q_norm + action, min=-0.95, max=0.95)

        # Unnormalize and return
        q_next = unnormalize_franka_joints(q_next_norm).type_as(q)

        return q_next.squeeze().detach().cpu().numpy()

    @torch.no_grad()
    def check_success(self, q_current: np.ndarray, target_pose: SE3) -> bool:
        """
        Check if we've reached the target pose

        :param q_current np.ndarray: Current joint configuration
        :param target_pose SE3: Target pose
        :rtype bool: Whether we've reached the target
        """
        # Get current end-effector pose
        eff_pose = FrankaRealRobot.fk(q_current, eff_frame="right_gripper")

        # Check position and orientation thresholds
        position_error = np.linalg.norm(eff_pose._xyz - target_pose._xyz)
        orientation_error = np.abs(
            np.degrees((eff_pose.so3._quat * target_pose.so3._quat.conjugate).radians)
        )

        return position_error < 0.01 and orientation_error < 15


class ReactiveControllerNode:
    def __init__(self):
        """
        Initializes the reactive controller node
        """
        rospy.init_node("mpinets_reactive_controller")
        time.sleep(1)

        self.controller = None
        self.base_frame = "panda_link0"

        # Control parameters
        self.control_rate = rospy.get_param("~control_rate", 10.0)  # Hz
        self.control_dt = 1.0 / self.control_rate

        # Log feedback mode
        rospy.loginfo(
            f"Joint state feedback mode: {'Self-feedback (simulation)' if USE_SELF_FEEDBACK else 'Real robot feedback'}"
        )

        # State variables - Initialize based on feedback mode
        if USE_SELF_FEEDBACK:
            # Self-feedback mode: start with neutral config
            self.current_joint_state = NEUTRAL_CONFIG[:7].copy()  # Only arm joints
            self.full_joint_state = NEUTRAL_CONFIG.copy()  # Full state for publishing
            rospy.loginfo(
                "Initialized with neutral configuration for self-feedback mode"
            )
        else:
            # Real robot mode: wait for real joint states
            self.current_joint_state = None
            self.full_joint_state = None
            rospy.loginfo("Waiting for real joint states...")

        self.target_pose = None
        self.latest_pointcloud = None
        self.is_controlling = False
        self.control_thread = None
        self.control_lock = threading.Lock()

        # Get the point cloud path parameter
        point_cloud_path = rospy.get_param("~point_cloud_path", "")

        # Determine mode based on whether point_cloud_path is provided and valid
        self.use_live_pointcloud = True  # Default to live mode

        if point_cloud_path and os.path.exists(point_cloud_path):
            self.use_live_pointcloud = False
            rospy.loginfo(f"Using file pointcloud mode: {point_cloud_path}")
            self.load_point_cloud_data(point_cloud_path)
        else:
            if point_cloud_path:
                rospy.logwarn(
                    f"Point cloud file not found: {point_cloud_path}. Switching to live mode."
                )
            rospy.loginfo("Using live pointcloud mode")

        # Publishers and subscribers
        self.joint_command_publisher = rospy.Publisher(
            "/position_joint_trajectory_controller/command",
            JointTrajectory,
            queue_size=1,
        )

        self.status_publisher = rospy.Publisher(
            "/mpinets/control_status", Bool, queue_size=1
        )

        self.full_point_cloud_publisher = rospy.Publisher(
            "/mpinets/full_point_cloud", PointCloud2, queue_size=1
        )

        # Joint state handling based on feedback mode
        if USE_SELF_FEEDBACK:
            # Self-feedback: publish our own joint states for visualization
            self.joint_state_publisher = rospy.Publisher(
                "/joint_states", JointState, queue_size=1
            )
            # Timer to publish joint states at high rate
            rospy.Timer(rospy.Duration(0.01), self.publish_joint_states)  # 100Hz
        else:
            # Real robot: subscribe to external joint states
            self.joint_state_subscriber = rospy.Subscriber(
                "/joint_states", JointState, self.joint_state_callback, queue_size=1
            )

        # Subscribe to planning problems (start/stop commands)
        self.planning_problem_subscriber = rospy.Subscriber(
            "/mpinets/planning_problem",
            PlanningProblem,
            self.planning_problem_callback,
            queue_size=1,
        )

        if self.use_live_pointcloud:
            # Subscribe to PRE-PROCESSED point cloud topic
            self.processed_pointcloud_subscriber = rospy.Subscriber(
                "/mpinets/processed_pointcloud",
                PointCloud2,
                self.processed_pointcloud_callback,
                queue_size=1,
            )
            rospy.loginfo(
                "Waiting for pre-processed pointcloud data from /mpinets/processed_pointcloud..."
            )

            # Start a timer to publish the pointcloud for visualization
            rospy.Timer(rospy.Duration(1.0), self.publish_pointcloud_data)

        # Load model
        rospy.loginfo("Loading model")
        self.controller = ReactiveController(rospy.get_param("~mdl_path"))
        rospy.loginfo("Model loaded")
        rospy.loginfo("Reactive controller ready")

    def publish_joint_states(self, event=None):
        """
        Publish current joint states for visualization (self-feedback mode only)
        """
        if not USE_SELF_FEEDBACK or self.full_joint_state is None:
            return

        with self.control_lock:
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "panda_link0"
            msg.name = [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]
            msg.position = self.full_joint_state.tolist()
            msg.velocity = [0.0] * 9
            msg.effort = [0.0] * 9

            self.joint_state_publisher.publish(msg)

    def joint_state_callback(self, msg: JointState):
        """
        Update current joint state from real robot (real robot mode only)
        """
        if USE_SELF_FEEDBACK:
            return  # Ignore real joint states in self-feedback mode

        with self.control_lock:
            if len(msg.position) >= 7:
                self.current_joint_state = np.array(msg.position[:7])
                if len(msg.position) >= 9:
                    self.full_joint_state = np.array(msg.position[:9])
                else:
                    # Pad with finger joint positions if not available
                    self.full_joint_state = np.concatenate(
                        [msg.position[:7], NEUTRAL_CONFIG[7:9]]
                    )

                if self.current_joint_state is not None:
                    rospy.loginfo_once("Received first real joint state")

    def processed_pointcloud_callback(self, msg: PointCloud2):
        """
        Callback for pre-processed pointcloud messages
        """
        try:
            points_list = []
            colors_list = []

            for p in pc2.read_points(
                msg, field_names=("x", "y", "z", "r", "g", "b", "a"), skip_nans=True
            ):
                points_list.append([p[0], p[1], p[2]])
                colors_list.append([p[3], p[4], p[5], p[6]])

            if points_list:
                points = np.array(points_list, dtype=np.float32)
                colors = np.array(colors_list, dtype=np.float32)

                if len(points) == NUM_OBSTACLE_POINTS:
                    with self.control_lock:
                        self.latest_pointcloud = points
                        self.latest_pointcloud_colors = colors
                    rospy.loginfo_once("Received first pre-processed point cloud")
                else:
                    rospy.logwarn_throttle(
                        10,
                        f"Pre-processed point cloud has {len(points)} points, expected {NUM_OBSTACLE_POINTS}",
                    )

        except Exception as e:
            rospy.logerr_throttle(10, f"Error reading processed point cloud: {e}")

    def load_point_cloud_data(self, path: str):
        """
        Load point cloud from file (similar to planning_node.py)
        """
        observation_data = np.load(path, allow_pickle=True).item()

        full_pc = tra.transform_points(
            observation_data["pc"], observation_data["camera_pose"]
        )

        no_robot_mask = (
            observation_data["label_map"]["robot"] != observation_data["pc_label"]
        )
        scene_pc = full_pc[no_robot_mask]

        scene_colors = observation_data["pc_color"][no_robot_mask] / 255.0
        scene_colors = np.concatenate(
            (scene_colors, np.ones((len(scene_colors), 1))), axis=1
        )

        # Clean and downsample
        scene_pc, scene_colors = self.clean_point_cloud(scene_pc, scene_colors)

        with self.control_lock:
            self.latest_pointcloud = scene_pc
            self.latest_pointcloud_colors = scene_colors

        rospy.Timer(
            rospy.Duration(1.0),
            partial(self.publish_point_cloud_data, scene_pc, scene_colors),
        )

    @staticmethod
    def clean_point_cloud(xyz: np.ndarray, rgba: np.ndarray):
        """
        Clean and downsample point cloud (same as planning_node.py)
        """
        if len(xyz) == 0:
            return np.zeros((NUM_OBSTACLE_POINTS, 3), dtype=np.float32), np.zeros(
                (NUM_OBSTACLE_POINTS, 4), dtype=np.float32
            )

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
            indices = np.random.choice(
                n_filtered, size=NUM_OBSTACLE_POINTS, replace=False
            )
            return xyz_filtered[indices].astype(np.float32), rgba_filtered[
                indices
            ].astype(np.float32)
        elif n_filtered > 0:
            repeat_factor = (NUM_OBSTACLE_POINTS + n_filtered - 1) // n_filtered
            xyz_repeated = np.repeat(xyz_filtered, repeat_factor, axis=0)
            rgba_repeated = np.repeat(rgba_filtered, repeat_factor, axis=0)

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

    def publish_pointcloud_data(self, event=None):
        """
        Publish pointcloud for visualization
        """
        with self.control_lock:
            if (
                self.latest_pointcloud is not None
                and hasattr(self, "latest_pointcloud_colors")
                and self.latest_pointcloud_colors is not None
            ):
                self.publish_point_cloud_data(
                    self.latest_pointcloud, self.latest_pointcloud_colors
                )

    def publish_point_cloud_data(self, points: np.ndarray, colors: np.ndarray, _=None):
        """
        Publish point cloud for visualization (same as planning_node.py)
        """
        if len(points) == 0:
            return

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

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

    def planning_problem_callback(self, msg: PlanningProblem):
        """
        Start/restart reactive control with new target
        """
        if self.controller is None:
            rospy.logwarn("Controller is not yet loaded")
            return

        # Extract target pose
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

        with self.control_lock:
            self.target_pose = target

            # Stop current control if running
            if self.is_controlling:
                self.is_controlling = False
                if self.control_thread and self.control_thread.is_alive():
                    self.control_thread.join()

            # Start new control thread
            self.is_controlling = True
            self.control_thread = threading.Thread(target=self.control_loop)
            self.control_thread.start()

        rospy.loginfo("Started reactive control with new target")

    def control_loop(self):
        """
        Main reactive control loop
        """
        rate = rospy.Rate(self.control_rate)

        while self.is_controlling and not rospy.is_shutdown():
            try:
                with self.control_lock:
                    # Check if we have all necessary data
                    if (
                        self.current_joint_state is None
                        or self.target_pose is None
                        or self.latest_pointcloud is None
                    ):
                        # Log waiting status based on mode
                        if not USE_SELF_FEEDBACK and self.current_joint_state is None:
                            rospy.logwarn_throttle(
                                5, "Waiting for real joint states..."
                            )
                        elif self.target_pose is None:
                            rospy.logwarn_throttle(5, "Waiting for target pose...")
                        elif self.latest_pointcloud is None:
                            rospy.logwarn_throttle(5, "Waiting for point cloud data...")
                        continue

                    # Check if we've reached the target
                    if self.controller.check_success(
                        self.current_joint_state, self.target_pose
                    ):
                        rospy.loginfo("Target reached! Stopping reactive control.")
                        self.is_controlling = False
                        self.status_publisher.publish(Bool(data=False))
                        break

                    # Get next action
                    q_next = self.controller.get_next_action(
                        self.current_joint_state,
                        self.target_pose,
                        self.latest_pointcloud,
                    )

                    # Update internal state based on feedback mode
                    if USE_SELF_FEEDBACK:
                        # Self-feedback: use commanded positions as current state
                        self.current_joint_state = q_next.copy()
                        self.full_joint_state[:7] = q_next
                        # Note: Joint states will be published by the timer

                # Send joint command
                self.send_joint_command(q_next)
                self.status_publisher.publish(Bool(data=True))

            except Exception as e:
                rospy.logerr(f"Error in control loop: {e}")
                self.is_controlling = False
                self.status_publisher.publish(Bool(data=False))
                break

            rate.sleep()

        rospy.loginfo("Reactive control loop stopped")

    def send_joint_command(self, joint_positions: np.ndarray):
        """
        Send joint command to the robot
        """
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = rospy.Time.now()
        traj_msg.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.velocities = [0.0] * 7
        point.accelerations = [0.0] * 7
        point.time_from_start = rospy.Duration.from_sec(self.control_dt)

        traj_msg.points.append(point)
        self.joint_command_publisher.publish(traj_msg)


if __name__ == "__main__":
    ReactiveControllerNode()
    rospy.spin()
