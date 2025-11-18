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
NEUTRAL_TARGET_XYZ = [0.40649957127333377, -0.4007287351995245575, 0.1066376674460814]
NEUTRAL_TARGET_XYZW = [-0.99965734, -0.01424194, -0.02026548, -0.00846602]

# Second target position (slightly offset from neutral)
SECOND_TARGET_XYZ = [0.40649957127333377, 0.407287351995245575, 0.1066376674460814]
SECOND_TARGET_XYZW = [-0.99965734, -0.01424194, -0.02026548, -0.00846602]

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

        # State variables
        self.target_a_xyz = NEUTRAL_TARGET_XYZ
        self.target_a_xyzw = NEUTRAL_TARGET_XYZW
        self.target_b_xyz = SECOND_TARGET_XYZ
        self.target_b_xyzw = SECOND_TARGET_XYZW
        self.current_target = 'A'  # Current target in sequence
        self.current_joint_state = NEUTRAL_CONFIG.tolist()
        self.is_controlling = False
        self.sequence_mode = False  # Whether we're running a sequence
        
        # NEW: Cycle control variables
        self.num_cycles = 3  # Number of back-and-forth cycles
        self.current_cycle = 0
        self.moving_forward = True  # True: A->B, False: B->A

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
        
        # NEW: Get number of cycles from parameter server
        self.num_cycles = rospy.get_param("~num_cycles", 3)
        # rospy.loginfo(f"Configured for {self.num_cycles} back-and-forth cycles")

        # Create target markers
        self.make_target_marker(self.target_a_xyz, self.target_a_xyzw, "target_a", "Target A", [0.0, 1.0, 0.0, 1.0])  # Green
        self.make_target_marker(self.target_b_xyz, self.target_b_xyzw, "target_b", "Target B", [1.0, 0.0, 0.0, 1.0])  # Red
        
        self.server.applyChanges()

    def control_status_callback(self, msg):
        """
        Update control status and handle sequence transitions
        """
        was_controlling = self.is_controlling
        self.is_controlling = msg.data
        
        # If we were controlling but now we're not, and we're in sequence mode, move to next target
        if was_controlling and not self.is_controlling and self.sequence_mode:
            # rospy.loginfo("Control completed for target {}".format(self.current_target))
            
            # NEW: Handle back-and-forth cycling logic
            if self.moving_forward:
                # We just completed A->B
                if self.current_target == 'A':
                    # Move to target B
                    self.current_target = 'B'
                    # rospy.loginfo(f"Cycle {self.current_cycle + 1}/{self.num_cycles}: Moving to target B")
                    time.sleep(0.5)  # Brief pause between targets
                    self.send_planning_problem(self.target_b_xyz, self.target_b_xyzw)
                elif self.current_target == 'B':
                    # Completed A->B, check if we need to go back
                    if self.current_cycle < self.num_cycles - 1:
                        # More cycles to go, reverse direction
                        self.moving_forward = False
                        self.current_target = 'A'
                        # rospy.loginfo(f"Cycle {self.current_cycle + 1}/{self.num_cycles}: Moving back to target A")
                        time.sleep(0.5)
                        self.send_planning_problem(self.target_a_xyz, self.target_a_xyzw)
                    else:
                        # Final cycle completed
                        # rospy.loginfo(f"Sequence complete: {self.num_cycles} back-and-forth cycles finished!")
                        self.sequence_mode = False
                        self.current_cycle = 0
                        self.moving_forward = True
                        self.current_target = 'A'
            else:
                # We just completed B->A
                if self.current_target == 'B':
                    # Move to target A
                    self.current_target = 'A'
                    # rospy.loginfo(f"Cycle {self.current_cycle + 1}/{self.num_cycles}: Moving to target A")
                    time.sleep(0.5)
                    self.send_planning_problem(self.target_a_xyz, self.target_a_xyzw)
                elif self.current_target == 'A':
                    # Completed B->A, increment cycle and reverse direction
                    self.current_cycle += 1
                    self.moving_forward = True
                    if self.current_cycle < self.num_cycles:
                        # Start next cycle: A->B
                        self.current_target = 'B'
                        # rospy.loginfo(f"Cycle {self.current_cycle + 1}/{self.num_cycles}: Starting next cycle to target B")
                        time.sleep(0.5)
                        self.send_planning_problem(self.target_b_xyz, self.target_b_xyzw)
                    else:
                        # All cycles completed
                        # rospy.loginfo(f"Sequence complete: {self.num_cycles} back-and-forth cycles finished!")
                        self.sequence_mode = False
                        self.current_cycle = 0
                        self.moving_forward = True
                        self.current_target = 'A'

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
        Creates a green cube that starts reactive control sequence (A -> B -> A -> B ... for n cycles)
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

    def make_gripper_control(self, msg, color=None):
        """
        Creates the gripper marker for the target
        """
        control = InteractiveMarkerControl()
        control.always_visible = True
        gripper_marker = self.make_gripper(msg)
        if color:
            gripper_marker.color.r, gripper_marker.color.g, gripper_marker.color.b, gripper_marker.color.a = color
        control.markers.append(gripper_marker)
        msg.controls.append(control)
        return msg.controls[-1]

    def make_target_marker(self, xyz, xyzw, name, description, color=None):
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
        int_marker.name = name
        int_marker.description = description

        self.make_gripper_control(int_marker, color)
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
        
        # Set callback based on target name
        if name == "target_a":
            self.server.setCallback(int_marker.name, self.target_a_feedback)
        elif name == "target_b":
            self.server.setCallback(int_marker.name, self.target_b_feedback)

    def send_planning_problem(self, target_xyz, target_xyzw):
        """
        Helper method to send planning problem
        """
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
        ) = target_xyz
        (
            msg.target.transform.rotation.x,
            msg.target.transform.rotation.y,
            msg.target.transform.rotation.z,
            msg.target.transform.rotation.w,
        ) = target_xyzw
        msg.q0 = JointState(position=self.current_joint_state[:7])

        self.planning_problem_publisher.publish(msg)

    def start_button_callback(self, feedback):
        """
        Start reactive control sequence (A -> B -> A -> B ... for n cycles)
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            # rospy.loginfo(f"Starting back-and-forth control: {self.num_cycles} cycles (A <-> B)")
            self.sequence_mode = True
            self.current_cycle = 0
            self.moving_forward = True
            self.current_target = 'A'
            
            # Start with target A
            self.send_planning_problem(self.target_a_xyz, self.target_a_xyzw)

        self.server.applyChanges()

    def stop_button_callback(self, feedback):
        """
        Stop reactive control - actually send a stop command now
        """
        if feedback.event_type == InteractiveMarkerFeedback.BUTTON_CLICK:
            # rospy.loginfo("Stopping reactive control and resetting sequence")
            # Publish stop command
            self.stop_control_publisher.publish(Bool(data=True))
            self.sequence_mode = False
            self.current_cycle = 0
            self.moving_forward = True
            self.current_target = 'A'  # Reset sequence to start

        self.server.applyChanges()

    def target_a_feedback(self, feedback):
        """
        Update target A pose when user moves the marker
        """
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.target_a_xyz = (
                feedback.pose.position.x,
                feedback.pose.position.y,
                feedback.pose.position.z,
            )
            self.target_a_xyzw = (
                feedback.pose.orientation.x,
                feedback.pose.orientation.y,
                feedback.pose.orientation.z,
                feedback.pose.orientation.w,
            )

    def target_b_feedback(self, feedback):
        """
        Update target B pose when user moves the marker
        """
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.target_b_xyz = (
                feedback.pose.position.x,
                feedback.pose.position.y,
                feedback.pose.position.z,
            )
            self.target_b_xyzw = (
                feedback.pose.orientation.x,
                feedback.pose.orientation.y,
                feedback.pose.orientation.z,
                feedback.pose.orientation.w,
            )


if __name__ == "__main__":
    env = ReactiveInterface()
    rospy.spin()