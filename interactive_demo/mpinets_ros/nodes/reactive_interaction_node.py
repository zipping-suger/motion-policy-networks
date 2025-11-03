#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import TransformStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import (
    Marker,
    InteractiveMarkerFeedback,
    InteractiveMarker,
    InteractiveMarkerControl,
)
from sensor_msgs.msg import PointCloud2, PointField, JointState
from mpinets_msgs.msg import PlanningProblem
import numpy as np
import time
import tf2_ros
from copy import deepcopy

# The neutral configuration at which to start the node
NEUTRAL_CONFIG = np.array([
    -0.01779206,
    -0.76012354,
    0.01978261,
    -2.34205014,
    0.02984053,
    1.54119353,
    0.75344866,
    0.025,
    0.025,
])

# A neutral starting target (matches the end effector of the neutral start)
NEUTRAL_TARGET_XYZ = [0.30649957127333377, 0.007287351995245575, 0.4866376674460814]
NEUTRAL_TARGET_XYZW = [-0.99965734, -0.01424194, -0.02026548, -0.00846602]

# The joint names
JOINT_NAMES = [
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


class ReactiveInterface:
    def __init__(self):
        """
        Initialize the system state, the interactive components, and the subscribers/publishers
        """
        rospy.init_node("mpinets_reactive_interface")
        self.server = InteractiveMarkerServer("mpinets_controls", "")
        self.br = tf2_ros.TransformBroadcaster()

        # UI elements
        self.make_start_button_marker([0.7, -1.0, 0.1], 0.2)
        self.make_stop_button_marker([1.0, -1.0, 0.1], 0.2)
        self.make_reset_button_marker([0.4, -1.0, 0.1], 0.2)

        # State variables
        self.target_xyz = NEUTRAL_TARGET_XYZ
        self.target_xyzw = NEUTRAL_TARGET_XYZW
        self.current_joint_state = NEUTRAL_CONFIG.tolist()
        self.is_controlling = False

        # Publishers
        self.planning_problem_publisher = rospy.Publisher(
            "/mpinets/planning_problem", PlanningProblem, queue_size=1
        )
        self.joint_states_publisher = rospy.Publisher(
            "/mpinets/joint_states", JointState, queue_size=1
        )
        # Add stop command publisher
        self.stop_control_publisher = rospy.Publisher(
            "/mpinets/stop_control", Bool, queue_size=1
        )

        # Subscribers
        self.real_joint_state_subscriber = rospy.Subscriber(
            "/joint_states", JointState, self.real_joint_state_callback, queue_size=1
        )
        self.control_status_subscriber = rospy.Subscriber(
            "/mpinets/control_status", Bool, self.control_status_callback, queue_size=1
        )

        # Create target marker
        self.make_target_marker(self.target_xyz, self.target_xyzw)
        self.server.applyChanges()

        time.sleep(1)
        self.reset_franka()

    def reset_franka(self):
        """
        Reset the robot to the neutral pose
        """
        rospy.loginfo("Resetting robot to neutral pose")

        # Publish neutral joint state
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.header.frame_id = "panda_link0"
        joint_msg.position = NEUTRAL_CONFIG.tolist()
        joint_msg.name = JOINT_NAMES
        self.current_joint_state = NEUTRAL_CONFIG.tolist()
        self.joint_states_publisher.publish(joint_msg)

    def control_status_callback(self, msg):
        """
        Update control status
        """
        self.is_controlling = msg.data

    def real_joint_state_callback(self, msg):
        """
        Callback to continuously update current joint state from real robot
        """
        self.current_joint_state = list(msg.position)

        # Republish for visualization
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.header.frame_id = "panda_link0"
        joint_msg.position = self.current_joint_state
        joint_msg.name = JOINT_NAMES
        self.joint_states_publisher.publish(joint_msg)

    @staticmethod
    def make_box(side_length, color):
        """
        Makes a colored box that can be viewed in Rviz
        """
        marker = Marker()
        marker.type = Marker.CUBE
        marker.scale.x = side_length
        marker.scale.y = side_length
        marker.scale.z = side_length
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    @staticmethod
    def make_gripper(msg):
        """
        Creates a floating gripper that can be viewed in Rviz
        """
        marker = Marker()
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = "package://mpinets_ros/meshes/half_open_gripper.stl"
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        return marker

    def make_start_button_marker(self, xyz, side_length):
        """
        Creates a green cube that starts reactive control
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5
        int_marker.name = "start_button"
        int_marker.description = "Start Control"

        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "start_button_control"
        marker = self.make_box(side_length, [45.0 / 255, 201.0 / 255, 55.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.start_button_callback)

    def make_stop_button_marker(self, xyz, side_length):
        """
        Creates a red cube that stops reactive control
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5
        int_marker.name = "stop_button"
        int_marker.description = "Stop Control"

        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "stop_button_control"
        marker = self.make_box(side_length, [204.0 / 255, 50.0 / 255, 50.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.stop_button_callback)

    def make_reset_button_marker(self, xyz, side_length):
        """
        Creates a yellow cube that resets the system
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        int_marker.scale = 0.5
        int_marker.name = "reset_button"
        int_marker.description = "Reset"

        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.name = "reset_button_control"
        marker = self.make_box(side_length, [231.0 / 255, 180.0 / 255, 22.0 / 255, 1.0])
        control.markers.append(marker)
        control.always_visible = True
        int_marker.controls.append(control)

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.reset_button_callback)

    def make_gripper_control(self, msg):
        """
        Creates the gripper marker for the target
        """
        control = InteractiveMarkerControl()
        control.always_visible = True
        control.markers.append(self.make_gripper(msg))
        msg.controls.append(control)
        return msg.controls[-1]

    def make_target_marker(self, xyz, xyzw):
        """
        Create the target interactive marker
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = "panda_link0"
        (
            int_marker.pose.position.x,
            int_marker.pose.position.y,
            int_marker.pose.position.z,
        ) = xyz
        (
            int_marker.pose.orientation.x,
            int_marker.pose.orientation.y,
            int_marker.pose.orientation.z,
            int_marker.pose.orientation.w,
        ) = xyzw
        int_marker.scale = 0.4
        int_marker.name = "target"
        int_marker.description = "Target Pose"

        self.make_gripper_control(int_marker)
        int_marker.controls[0].interaction_mode = InteractiveMarkerControl.NONE

        # Add 6DOF controls
        control = InteractiveMarkerControl()
        xyzw = np.array([1.0, 0.0, 0.0, 1.0])
        xyzw = xyzw / np.linalg.norm(xyzw)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = xyzw
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        xyzw = np.array([0.0, 1.0, 0.0, 1.0])
        xyzw = xyzw / np.linalg.norm(xyzw)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = xyzw
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        xyzw = np.array([0.0, 0.0, 1.0, 1.0])
        xyzw = xyzw / np.linalg.norm(xyzw)
        (
            control.orientation.x,
            control.orientation.y,
            control.orientation.z,
            control.orientation.w,
        ) = xyzw
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        int_marker.controls.append(deepcopy(control))
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(deepcopy(control))

        self.server.insert(int_marker)
        self.server.setCallback(int_marker.name, self.target_feedback)

    def start_button_callback(self, feedback):
        """
        Start reactive control
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo("Starting reactive control")

            msg = PlanningProblem()
            msg.header.stamp = rospy.Time.now()
            msg.joint_names = JOINT_NAMES[:7]
            msg.target = TransformStamped()
            msg.target.header.frame_id = "panda_link0"
            msg.target.child_frame_id = "target_frame"
            (
                msg.target.transform.translation.x,
                msg.target.transform.translation.y,
                msg.target.transform.translation.z,
            ) = self.target_xyz
            (
                msg.target.transform.rotation.x,
                msg.target.transform.rotation.y,
                msg.target.transform.rotation.z,
                msg.target.transform.rotation.w,
            ) = self.target_xyzw
            msg.q0 = JointState(position=self.current_joint_state[:7])

            self.planning_problem_publisher.publish(msg)

        self.server.applyChanges()

    def stop_button_callback(self, feedback):
        """
        Stop reactive control - actually send a stop command now
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo("Stopping reactive control")
            # Publish stop command
            self.stop_control_publisher.publish(Bool(data=True))

        self.server.applyChanges()

    def reset_button_callback(self, feedback):
        """
        Reset system
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            rospy.loginfo("Resetting system")
            self.reset_franka()

        self.server.applyChanges()

    def target_feedback(self, feedback):
        """
        Update target pose when user moves the marker
        """
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.target_xyz = (
                feedback.pose.position.x,
                feedback.pose.position.y,
                feedback.pose.position.z,
            )
            self.target_xyzw = (
                feedback.pose.orientation.x,
                feedback.pose.orientation.y,
                feedback.pose.orientation.z,
                feedback.pose.orientation.w,
            )


if __name__ == "__main__":
    env = ReactiveInterface()
    rospy.spin()