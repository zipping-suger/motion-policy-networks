import numpy as np
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
import json
from datetime import datetime
import random

from robofin.robots import FrankaRobot, FrankaGripper
from robofin.bullet import BulletController, Bullet
from robofin.pointcloud.torch import FrankaSampler

from mpinets.model import MotionPolicyNetwork
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from mpinets.geometry import construct_mixed_point_cloud

# Import the metrics module
from mpinets.metrics import Evaluator

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 75
GOAL_THRESHOLD = 0.01  # 1 cm threshold for goal reaching


class DynamicObstacleExperiment:
    def __init__(self, mdl_path, gui=False, z_min=0.2, z_max=0.8):
        # Load MotionPolicyNetwork
        self.model = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
        self.model.eval()

        self.cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
        self.gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
        
        # Initialize simulation
        self.sim = BulletController(hz=8, substeps=80, gui=gui)  # Set gui based on parameter
        self.franka = self.sim.load_robot(FrankaRobot)
        
        # Initialize collision detection simulation
        self.collision_sim = Bullet(gui=False)
        self.collision_robot = self.collision_sim.load_robot(FrankaRobot)
        
        # Set camera
        self.sim.set_camera_position(yaw=-90, pitch=-30, distance=2.5, target=[0.0, 0.0, 0.5])
        
        # Define two target poses for the robot to alternate between using midpoint RPY
        p1 = [0.3, 0.4, 0.5]
        roll1 = np.pi
        pitch1 = 0
        yaw1 = 0
        self.target1 = SE3(xyz=p1, so3=SO3.from_rpy(roll1, pitch1, yaw1))

        p2 = [0.3, -0.4, 0.5]
        roll2 = np.pi
        pitch2 = 0
        yaw2 = 0
        self.target2 = SE3(xyz=p2, so3=SO3.from_rpy(roll2, pitch2, yaw2))
        
        self.current_target = self.target1
        
        # Create visual markers for targets
        self.target_gripper1 = self.sim.load_robot(FrankaGripper, collision_free=True)
        self.target_gripper2 = self.sim.load_robot(FrankaGripper, collision_free=True)
        self.target_gripper1.marionette(self.target1)
        self.target_gripper2.marionette(self.target2)
        
        # Define obstacle randomization parameters
        self.obstacle_z_min = z_min  # Minimum Z position (meters)
        self.obstacle_z_max = z_max  # Maximum Z position (meters)
        
        # Create a dynamic obstacle (cuboid) with identity quaternion
        # Randomize initial Z position
        random_z = random.uniform(self.obstacle_z_min, self.obstacle_z_max)
        self.obstacle = Cuboid(center=[0.3, 0.0, random_z], dims=[0.2, 0.05, 0.2], quaternion=[1, 0, 0, 0])
        # Load the obstacle and store its ID
        self.obstacle_ids = self.sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
        self.obstacle_id = self.obstacle_ids[0]  # Get the first (and only) obstacle ID
        
        # Also load obstacle in collision simulation
        self.collision_obstacle_ids = self.collision_sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
        
        # Start with the robot at target1
        self.franka.marionette(self.get_config_for_target(self.target2))
        self.collision_robot.marionette(self.get_config_for_target(self.target2))
        
        # Control variables
        self.obstacle_velocity = np.array([0.0, 0.0, 0.0])
        self.running = True
        self.obstacle_direction = 1  # 1 for up, -1 for down
        self.obstacle_amplitude = 0.3  # Movement amplitude in meters
        self.obstacle_center_z = random_z  # Use the randomized value instead of fixed 0.5
        
        # Metrics collection
        self.metrics = {
            'success_rate': [],
            'avg_time_to_target': [],
            'collision_count': [],
            'avg_path_length': [],
            'speed_values': []
        }
        
        # Experiment parameters
        self.speeds_to_test = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
        # self.speeds_to_test = [0.2, 0.3, 0.4, 0.5] # For quicker testing
        self.trials_per_speed = 50
        self.max_trial_time = 30.0  # seconds
           
    def get_config_for_target(self, target):
        # # Use the robot's random_ik method to get a proper joint configuration
        # # Try multiple times to find a valid solution
        # max_attempts = 100
        
        # for attempt in range(max_attempts):
        #     try:
        #         # Get random IK solutions for the target pose (using right_gripper frame)
        #         solutions = FrankaRobot.random_ik(target, eff_frame="right_gripper")
                
        #         if len(solutions) > 0:
        #             # Choose the first valid solution
        #             print("solution found", solutions[0])
        #             return solutions[0]
        #     except Exception as e:
        #         # Continue trying if IK fails
        #         continue
        # # If all attempts fail, use a fallback configuration
        # print("No IK solution found after 100 attempts, using default configuration")  
              
        base_config = np.array([-1.9591217,   1.41348107,  2.32425933, -1.71939206, -1.23057707,  0.86966958, -2.40812116])
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
    
    def update_obstacle_position(self, speed, dt):
        # Calculate new z position with sinusoidal movement
        max_displacement = speed * dt
        new_z = self.obstacle.center[2] + self.obstacle_direction * max_displacement
        
        # Reverse direction if at amplitude limits
        if new_z > self.obstacle_center_z + self.obstacle_amplitude:
            new_z = self.obstacle_center_z + self.obstacle_amplitude
            self.obstacle_direction = -1
        elif new_z < self.obstacle_center_z - self.obstacle_amplitude:
            new_z = self.obstacle_center_z - self.obstacle_amplitude
            self.obstacle_direction = 1
            
        # Update obstacle position
        center = [self.obstacle.center[0], self.obstacle.center[1], new_z]
        self.obstacle.center = center
        
        # Update the obstacle in simulation
        self.sim.clear_all_obstacles()
        self.obstacle_ids = self.sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
        self.obstacle_id = self.obstacle_ids[0]
        
        # Also update obstacle in collision simulation
        self.collision_sim.clear_all_obstacles()
        self.collision_obstacle_ids = self.collision_sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
    
    def check_collision(self, config):
        # Update collision robot to current configuration
        self.collision_robot.marionette(config)
        
        # Check for collisions between robot and obstacle using the collision simulation
        return self.collision_sim.in_collision(self.collision_robot, check_self=False)
    
    def run_trial(self, speed, trial_idx):
        print(f"Running trial {trial_idx+1} with obstacle speed: {speed} m/s")
        
        # Randomize obstacle starting Z position for this trial
        random_z = random.uniform(self.obstacle_z_min, self.obstacle_z_max)
        print(f"  Randomized obstacle Z position: {random_z:.3f} meters")
        
        # Reset environment with randomized position
        self.obstacle.center = [0.3, 0.0, random_z]
        self.obstacle_center_z = random_z  # Update the center reference
        
        self.sim.clear_all_obstacles()
        self.obstacle_ids = self.sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
        self.obstacle_id = self.obstacle_ids[0]
        
        # Reset collision simulation
        self.collision_sim.clear_all_obstacles()
        self.collision_obstacle_ids = self.collision_sim.load_primitives([self.obstacle], color=[0.8, 0.2, 0.2, 1])
        
        # Reset robot to initial position
        initial_config = self.get_config_for_target(self.target2)
        self.franka.marionette(initial_config)
        self.collision_robot.marionette(initial_config)
        self.current_target = self.target1
        
        # Precompute obstacle points
        obstacle_points = construct_mixed_point_cloud([self.obstacle], NUM_OBSTACLE_POINTS)
        obstacle_points_tensor = torch.tensor(
            obstacle_points[:, :3], dtype=torch.float32, device="cuda:0"
        )
        
        # Trial metrics
        success = False
        collision = False
        start_time = time.time()
        path_length = 0
        last_position = None
        
        # Run trial
        while time.time() - start_time < self.max_trial_time:
            # Update obstacle position
            dt = 1/8  # Simulation frequency is 8Hz
            self.update_obstacle_position(speed, dt)
            
            # Update obstacle points based on current position
            obstacle_points = construct_mixed_point_cloud([self.obstacle], NUM_OBSTACLE_POINTS)
            obstacle_points_tensor = torch.tensor(
                obstacle_points[:, :3], dtype=torch.float32, device="cuda:0"
            )
            
            # Get current robot configuration
            current_config, _ = self.franka.get_joint_states()
            current_config = current_config[:7]  # Only arm joints
            
            # Track path length
            current_position = FrankaRobot.fk(current_config).xyz
            if last_position is not None:
                path_length += np.linalg.norm(np.array(current_position) - np.array(last_position))
            last_position = current_position
            
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
            
            # Check for collision
            if self.check_collision(next_config):
                collision = True
                break
            
            # Check if we reached the current target
            current_ee = FrankaRobot.fk(next_config).xyz
            distance = np.linalg.norm(
                np.array(current_ee) - np.array(self.current_target.xyz)
            )
            
            if distance < GOAL_THRESHOLD:
                success = True
                break
        
        trial_time = time.time() - start_time
                
        return {
            'success': success,
            'collision': collision,
            'time': trial_time if success else self.max_trial_time,
            'path_length': path_length
        }
    
    def run_experiment(self):
        print("Starting dynamic obstacle experiment...")
        print(f"Obstacle Z position randomization range: [{self.obstacle_z_min}, {self.obstacle_z_max}] meters")
        
        for speed in self.speeds_to_test:
            successes = 0
            collisions = 0
            total_time = 0
            total_path_length = 0
            
            for trial in range(self.trials_per_speed):
                results = self.run_trial(speed, trial)
                
                if results['success']:
                    successes += 1
                    total_time += results['time']
                    total_path_length += results['path_length']
                elif results['collision']:
                    collisions += 1
            
            # Calculate metrics for this speed
            success_rate = successes / self.trials_per_speed
            avg_time = total_time / successes if successes > 0 else 0
            avg_path_length = total_path_length / successes if successes > 0 else 0
            
            # Store metrics
            self.metrics['speed_values'].append(speed)
            self.metrics['success_rate'].append(success_rate)
            self.metrics['avg_time_to_target'].append(avg_time)
            self.metrics['collision_count'].append(collisions)
            self.metrics['avg_path_length'].append(avg_path_length)
            
            print(f"Speed: {speed} m/s")
            print(f"  Success rate: {success_rate*100:.1f}%")
            print(f"  Average time to target: {avg_time:.2f}s")
            print(f"  Collisions: {collisions}")
            print(f"  Average path length: {avg_path_length:.2f}m")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"obstacle_speed_experiment_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"Experiment completed. Results saved to {filename}")

        # Generate and save a human-readable summary
        summary_filename = f"obstacle_speed_experiment_summary_{timestamp}.txt"
        summary_lines = ["Experiment Summary\n", "--------------------\n"]
        summary_lines.append(f"Obstacle Z randomization range: [{self.obstacle_z_min}, {self.obstacle_z_max}] meters\n")
        
        for i, speed in enumerate(self.metrics['speed_values']):
            summary_lines.append(f"Speed: {speed} m/s")
            summary_lines.append(f"  Success rate: {self.metrics['success_rate'][i]*100:.1f}%")
            summary_lines.append(f"  Average time to target: {self.metrics['avg_time_to_target'][i]:.2f}s")
            summary_lines.append(f"  Collisions: {self.metrics['collision_count'][i]}")
            summary_lines.append(f"  Average path length: {self.metrics['avg_path_length'][i]:.2f}m\n")
            
        summary_text = "\n".join(summary_lines)
        
        with open(summary_filename, 'w') as f:
            f.write(summary_text)
            
        print(f"Summary saved to {summary_filename}")
        print("\n" + summary_text)
        
        return self.metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mdl_path", type=str, help="A checkpoint file from training MotionPolicyNetwork"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Enable PyBullet GUI for visualization"
    )
    parser.add_argument(
        "--z-min", type=float, default=0.2, help="Minimum Z position for obstacle randomization"
    )
    parser.add_argument(
        "--z-max", type=float, default=0.8, help="Maximum Z position for obstacle randomization"
    )
    args = parser.parse_args()
    
    experiment = DynamicObstacleExperiment(
        args.mdl_path, 
        gui=args.gui, 
        z_min=args.z_min,
        z_max=args.z_max
    )
    
    try:
        metrics = experiment.run_experiment()
    except KeyboardInterrupt:
        print("Experiment interrupted by user.")