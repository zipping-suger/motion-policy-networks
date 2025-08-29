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
import cv2
import time
import torch
from tqdm.auto import tqdm
from pathlib import Path
from geometrout.transform import SE3, SO3
from geometrout.primitive import Cuboid
from pyquaternion import Quaternion
import argparse
import pickle
import threading

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController
from robofin.pointcloud.torch import FrankaSampler

from mpinets.model import MotionPolicyNetwork
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets.geometry import construct_mixed_point_cloud

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75
GOAL_THRESHOLD = 0.01  # 1 cm threshold for goal reaching

class DynamicObstacleDemo:
    def __init__(self, mdl_path):
        # Load MotionPolicyNetwork
        self.model = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
        self.model.eval()

        self.cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
        self.gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
        
        # Initialize simulation
        self.sim = BulletController(hz=8, substeps=60, gui=True)
        self.franka = self.sim.load_robot(FrankaRobot)
        
        # Set camera
        self.sim.set_camera_position(yaw=-90, pitch=-30, distance=2.5, target=[0.0, 0.0, 0.5])
        
        # Define two target poses for the robot to alternate between using midpoint RPY
        p1 = [0.3, 0.4, 0.5]
        roll1 = (3 * np.pi / 4 + 5 * np.pi / 4) / 2
        pitch1 = (-np.pi / 8 + np.pi / 8) / 2
        yaw1 = (-np.pi / 2 + np.pi / 2) / 2
        self.target1 = SE3(xyz=p1, so3=SO3.from_rpy(roll1, pitch1, yaw1))

        p2 = [0.3, -0.4, 0.5]
        roll2 = (3 * np.pi / 4 + 5 * np.pi / 4) / 2
        pitch2 = (-np.pi / 8 + np.pi / 8) / 2
        yaw2 = (-np.pi / 2 + np.pi / 2) / 2
        self.target2 = SE3(xyz=p2, so3=SO3.from_rpy(roll2, pitch2, yaw2))
        
        self.current_target = self.target1
        
        # Create visual markers for targets
        self.target_gripper1 = self.sim.load_robot(FrankaGripper, collision_free=True)
        self.target_gripper2 = self.sim.load_robot(FrankaGripper, collision_free=True)
        self.target_gripper1.marionette(self.target1)
        self.target_gripper2.marionette(self.target2)
        
        # Create a dynamic obstacle (cuboid) with identity quaternion
        self.obstacle = Cuboid(center=[0.3, 0.0, 0.25], dims=[0.2, 0.05, 0.2], quaternion=[1, 0, 0, 0])
        # Load the obstacle and store its ID
        self.obstacle_ids = self.sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
        self.obstacle_id = self.obstacle_ids[0]  # Get the first (and only) obstacle ID
        
        # Start with the robot at target1
        self.franka.marionette(self.get_config_for_target(self.target1))
        
        # Control variables
        self.obstacle_velocity = np.array([0.0, 0.0, 0.0])
        self.running = True
        self.control_window_name = "Obstacle Control"
        
    def get_config_for_target(self, target):
        # Simple IK approximation - in a real scenario, you'd use proper IK
        # This is just a rough approximation for demonstration
        direction = target.xyz - np.array([0.0, 0.0, 0.5])
        direction = direction / np.linalg.norm(direction)
        
        # Create a configuration that points toward the target
        base_config = np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.0, 0.8])
        return base_config
    
    def create_point_cloud(self, robot_points, obstacle_points, target_points):
        pc = torch.zeros(
            NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS,
            4,  # x,y,z + segmentation mask
            device="cuda:0",
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
        mid_end = NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS
        pc[mid_end:, :3] = target_points
        pc[mid_end:, 3] = 2

        return pc.unsqueeze(0)  # Add batch dimension
    
    def move_obstacle_with_key(self, key, pos_step=0.05):
        moved = False
        center = np.array(self.obstacle.center)
        
        # Position changes
        if key == ord("w"):
            center = center + np.array([0, pos_step, 0])
            moved = True
        elif key == ord("s"):
            center = center + np.array([0, -pos_step, 0])
            moved = True
        elif key == ord("a"):
            center = center + np.array([-pos_step, 0, 0])
            moved = True
        elif key == ord("d"):
            center = center + np.array([pos_step, 0, 0])
            moved = True
        elif key == ord("q"):
            center = center + np.array([0, 0, pos_step])
            moved = True
        elif key == ord("e"):
            center = center + np.array([0, 0, -pos_step])
            moved = True
            
        if moved:
            self.obstacle.center = center
            # Update the obstacle in simulation using the proper method
            # Remove the old obstacle and create a new one at the new position
            self.sim.clear_all_obstacles()
            self.obstacle_ids = self.sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
            self.obstacle_id = self.obstacle_ids[0]
        return moved
    
    def control_obstacle(self):
        cv2.namedWindow(self.control_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window_name, 300, 150)
        
        control_img = np.zeros((150, 300, 3), dtype=np.uint8)
        cv2.putText(control_img, "WASD: Move XY", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, "Q/E: Move Z", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, "ESC: Exit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        while self.running:
            cv2.imshow(self.control_window_name, control_img)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                self.running = False
                break
            else:
                self.move_obstacle_with_key(key)
            time.sleep(0.03)
        
        cv2.destroyWindow(self.control_window_name)
    
    def run(self):
        # Start obstacle control thread
        control_thread = threading.Thread(target=self.control_obstacle)
        control_thread.daemon = True
        control_thread.start()
        
        print("Starting dynamic obstacle demo...")
        print("Use WASD (XY), QE (Z) to move the obstacle.")
        print("Press ESC in the control window to exit.")
        
        # Precompute obstacle points (we'll update the position in the loop)
        obstacle_points = construct_mixed_point_cloud([self.obstacle], NUM_OBSTACLE_POINTS)
        obstacle_points_tensor = torch.tensor(
            obstacle_points[:, :3], dtype=torch.float32, device="cuda:0"
        )
        
        while self.running:
            # Update obstacle points based on current position
            obstacle_points = construct_mixed_point_cloud([self.obstacle], NUM_OBSTACLE_POINTS)
            obstacle_points_tensor = torch.tensor(
                obstacle_points[:, :3], dtype=torch.float32, device="cuda:0"
            )
            
            # Get current robot configuration
            current_config, _ = self.franka.get_joint_states()
            current_config = current_config[:7]  # Only arm joints
            
            # Convert to tensor
            current_q = torch.tensor(
                current_config, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)
            q_norm = normalize_franka_joints(current_q)

            # Construct target points
            target_pose_mat = torch.tensor(
                self.current_target.matrix, dtype=torch.float32, device="cuda:0"
            ).unsqueeze(0)
            target_points = self.gpu_fk_sampler.sample_end_effector(
                target_pose_mat, NUM_TARGET_POINTS
            ).squeeze(0)

            # Construct the target pose input for the model
            target_position = torch.as_tensor(
                self.current_target.matrix[:3, 3], dtype=torch.float32
            )
            target_rot_mat = torch.as_tensor(
                self.current_target.matrix[:3, :3].flatten(), dtype=torch.float32
            )
            target_pose_input = (
                torch.cat((target_position, target_rot_mat), dim=0)
                .float()
                .unsqueeze(0)
                .to(q_norm.device)
            )

            # Sample robot points
            robot_points = self.gpu_fk_sampler.sample(
                current_q, NUM_ROBOT_POINTS
            ).squeeze(0)

            # Create point cloud
            xyz = self.create_point_cloud(
                robot_points, obstacle_points_tensor, target_points
            )

            # Policy prediction
            with torch.no_grad():
                delta_q = self.model(xyz, q_norm, target_pose_input)
                q_norm = torch.clamp(q_norm + delta_q, min=-1, max=1)
                next_q = unnormalize_franka_joints(q_norm)
                next_config = next_q.squeeze(0).detach().cpu().numpy()

            # Execute the next configuration
            self.franka.control_position(next_config)
            self.sim.step()
            time.sleep(0.05)
            
            # Check if we reached the current target
            current_ee = FrankaRobot.fk(next_config).xyz
            distance = np.linalg.norm(
                np.array(current_ee) - np.array(self.current_target.xyz)
            )
            
            if distance < GOAL_THRESHOLD:
                print(f"Reached target! Switching to next target.")
                # Switch targets
                if np.array_equal(self.current_target.xyz, self.target1.xyz):
                    self.current_target = self.target2
                else:
                    self.current_target = self.target1
        
        print("Exiting dynamic obstacle demo.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mdl_path", type=str, help="A checkpoint file from training MotionPolicyNetwork"
    )
    args = parser.parse_args()
    
    demo = DynamicObstacleDemo(args.mdl_path)
    try:
        demo.run()
    except KeyboardInterrupt:
        demo.running = False
        print("Demo interrupted by user.")