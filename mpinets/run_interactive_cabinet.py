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

# Import the cabinet environment
from environments.cabinet_environment import Cabinet

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75
GOAL_THRESHOLD = 0.02  # 1 cm threshold for goal reaching

class CabinetInteractiveDemo:
    def __init__(self, mdl_path):
        # Load MotionPolicyNetwork
        self.model = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
        self.model.eval()

        self.cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
        self.gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
        
        # Initialize simulation
        self.sim = BulletController(hz=5, substeps=100, gui=True)
        self.franka = self.sim.load_robot(FrankaRobot)
        
        # Set camera
        self.sim.set_camera_position(yaw=-90, pitch=-30, distance=2.5, target=[0.0, 0.0, 0.5])
        
        # Create cabinet
        self.cabinet = Cabinet()
        self.cabinet.cabinet_left = 0.35
        self.cabinet.cabinet_right = -0.35
        self.cabinet.cabinet_bottom = 0.2
        self.cabinet.cabinet_front = 0.4
        self.cabinet.cabinet_back = 0.9
        self.cabinet.cabinet_top = 1.0
        self.cabinet.thickness = 0.02
        self.cabinet.in_cabinet_rotation = 0
        self.cabinet.left_open_angle = np.pi/2  
        self.cabinet.right_open_angle = np.pi/4 
        
        # Load cabinet into simulation
        self.cabinet_cuboids = self.cabinet.cuboids
        self.cabinet_ids = self.sim.load_primitives(self.cabinet_cuboids, color=[0.7, 0.5, 0.3, 1])
        
        # Define target poses (inside and outside the cabinet)
        p_inside = [0.6, 0.0, 0.5]  # Inside the cabinet
        roll_inside = np.pi
        pitch_inside = 0
        yaw_inside = 0
        self.target_inside = SE3(xyz=p_inside, so3=SO3.from_rpy(roll_inside, pitch_inside, yaw_inside))
        
        # Use a neutral pose for the outside target
        neutral_config = np.array([
            -0.017792060227770554,
            -0.7601235411041661,
            0.019782607023391807,
            -2.342050140544315,
            0.029840531355804868,
            1.5411935298621688,
           0.7534486589746342,
        ])
        self.target_outside = FrankaRobot.fk(neutral_config, eff_frame="right_gripper")
        
        self.current_target = self.target_inside
        
        # Create visual markers for targets
        self.target_gripper_inside = self.sim.load_robot(FrankaGripper, collision_free=True)
        self.target_gripper_outside = self.sim.load_robot(FrankaGripper, collision_free=True)
        self.target_gripper_inside.marionette(self.target_inside)
        self.target_gripper_outside.marionette(self.target_outside)
        
        # Start with the robot at the neutral pose (outside target)
        self.franka.marionette(neutral_config)
        
        # Control variables
        self.running = True
        self.control_window_name = "Cabinet Door Control"
        
    def get_config_for_target(self, target):
        # Simple IK approximation
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
    
    def update_cabinet_doors(self, left_angle=None, right_angle=None):
        if left_angle is not None:
            self.cabinet.left_open_angle = left_angle
        if right_angle is not None:
            self.cabinet.right_open_angle = right_angle
        
        # Update cabinet in simulation
        self.sim.clear_all_obstacles()
        self.cabinet_cuboids = self.cabinet.cuboids
        self.cabinet_ids = self.sim.load_primitives(self.cabinet_cuboids, color=[0.7, 0.5, 0.3, 1])
    
    def control_cabinet_doors(self):
        cv2.namedWindow(self.control_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window_name, 400, 200)
        
        control_img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(control_img, "Q/A: Left Door +/-", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, "W/S: Right Door +/-", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, "T: Toggle Target", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(control_img, "ESC: Exit", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        angle_step = np.pi / 36  # 5 degrees
        
        while self.running:
            cv2.imshow(self.control_window_name, control_img)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                self.running = False
                break
            elif key == ord('q'):  # Increase left door angle
                new_angle = min(self.cabinet.left_open_angle + angle_step, np.pi)
                self.update_cabinet_doors(left_angle=new_angle)
            elif key == ord('a'):  # Decrease left door angle
                new_angle = max(self.cabinet.left_open_angle - angle_step, 0)
                self.update_cabinet_doors(left_angle=new_angle)
            elif key == ord('w'):  # Increase right door angle
                new_angle = min(self.cabinet.right_open_angle + angle_step, np.pi)
                self.update_cabinet_doors(right_angle=new_angle)
            elif key == ord('s'):  # Decrease right door angle
                new_angle = max(self.cabinet.right_open_angle - angle_step, 0)
                self.update_cabinet_doors(right_angle=new_angle)
            elif key == ord('t'):  # Toggle target
                if self.current_target == self.target_inside:
                    self.current_target = self.target_outside
                else:
                    self.current_target = self.target_inside
                print(f"Switched target to {'inside' if self.current_target == self.target_inside else 'outside'} the cabinet")
            
            time.sleep(0.03)
        
        cv2.destroyWindow(self.control_window_name)
    
    def run(self):
        # Start cabinet control thread
        control_thread = threading.Thread(target=self.control_cabinet_doors)
        control_thread.daemon = True
        control_thread.start()
        
        print("Starting cabinet interactive demo...")
        print("Use Q/A to control left door, W/S to control right door.")
        print("Press T to toggle between inside/outside targets.")
        print("Press ESC in the control window to exit.")
        
        # Precompute cabinet points
        cabinet_points = construct_mixed_point_cloud(self.cabinet_cuboids, NUM_OBSTACLE_POINTS)
        cabinet_points_tensor = torch.tensor(
            cabinet_points[:, :3], dtype=torch.float32, device="cuda:0"
        )
        
        while self.running:
            # Update cabinet points if doors have moved
            cabinet_points = construct_mixed_point_cloud(self.cabinet_cuboids, NUM_OBSTACLE_POINTS)
            cabinet_points_tensor = torch.tensor(
                cabinet_points[:, :3], dtype=torch.float32, device="cuda:0"
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
                robot_points, cabinet_points_tensor, target_points
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
            
            # Check if we reached the current target (just for feedback, not for switching)
            current_ee = FrankaRobot.fk(next_config).xyz
            distance = np.linalg.norm(
                np.array(current_ee) - np.array(self.current_target.xyz)
            )
            
            if distance < GOAL_THRESHOLD:
                print(f"Reached target at {self.current_target.xyz}!")
        
        print("Exiting cabinet interactive demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mdl_path", type=str, help="A checkpoint file from training MotionPolicyNetwork"
    )
    args = parser.parse_args()
    
    demo = CabinetInteractiveDemo(args.mdl_path)
    try:
        demo.run()
    except KeyboardInterrupt:
        demo.running = False
        print("Demo interrupted by user.")