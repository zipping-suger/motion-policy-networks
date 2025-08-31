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

from typing import Tuple
from mpinets import utils
from mpinets.geometry import TorchCuboids, TorchCylinders
import torch.nn.functional as F
import torch
from robofin.pointcloud.torch import FrankaSampler
from geometrout.primitive import Cuboid, Cylinder


def point_match_loss(input_pc: torch.Tensor, target_pc: torch.Tensor) -> torch.Tensor:
    """
    A combination L1 and L2 loss to penalize large and small deviations between
    two point clouds

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :rtype torch.Tensor: The single loss value
    """
    return F.mse_loss(input_pc, target_pc, reduction="mean") + F.l1_loss(
        input_pc, target_pc, reduction="mean"
    )


def collision_loss(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the hinge loss, calculating whether the robot (represented as a
    point cloud) is in collision with any obstacles in the scene. Collision
    here actually means within 3cm of the obstacle--this is to provide stronger
    gradient signal to encourage the robot to move out of the way. Also, some of the
    primitives can have zero volume (i.e. a dim is zero for cuboids or radius or height is zero for cylinders).
    If these are zero volume, they will have infinite sdf values (and therefore be ignored by the loss).

    :param input_pc torch.Tensor: Points sampled from the robot's surface after it
                                  is placed at the network's output prediction. Has dim [B, N, 3]
    :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
    :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
    :rtype torch.Tensor: Returns the loss value aggregated over the batch
    """

    cuboids = TorchCuboids(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
    )
    cylinders = TorchCylinders(
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
    )
    sdf_values = torch.minimum(cuboids.sdf(input_pc), cylinders.sdf(input_pc))
    return F.hinge_embedding_loss(
        sdf_values,
        -torch.ones_like(sdf_values),
        margin=0.03,
        reduction="mean",
    )


class CollisionAndBCLossContainer:
    """
    A container class to hold the various losses. This is structured as a
    container because that allows it to cache the robot pointcloud sampler
    object. By caching this, we reduce parsing time when processing the URDF
    and allow for a consistent random pointcloud (consistent per-GPU, that is)
    """

    def __init__(
        self,
    ):
        self.fk_sampler = None
        self.num_points = 1024

    def __call__(
        self,
        input_normalized: torch.Tensor,
        cuboid_centers: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_quaternions: torch.Tensor,
        cylinder_centers: torch.Tensor,
        cylinder_radii: torch.Tensor,
        cylinder_heights: torch.Tensor,
        cylinder_quaternions: torch.Tensor,
        target_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method calculates both constituent loss function after loading,
        and then caching, a fixed robot point cloud sampler (i.e. the task
        spaces sampled are always the same, as opposed to a random point cloud).
        The fixed point cloud is important for loss calculation so that
        it's possible to take mse between the two pointclouds.

        :param input_normalized torch.Tensor: Has dim [B, 7] and is always between -1 and 1
        :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
        :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
        :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
        :param target_normalized torch.Tensor: Has dim [B, 7] and is always between -1 and 1
        :rtype Tuple[torch.Tensor, torch.Tensor]: The two losses aggregated over the batch
        """
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(
                input_normalized.device,
                num_fixed_points=self.num_points,
                use_cache=True,
                with_base_link=False,  # Remove base link because this isn't controllable anyway
            )
        input_pc = self.fk_sampler.sample(
            utils.unnormalize_franka_joints(input_normalized),
        )
        target_pc = self.fk_sampler.sample(
            utils.unnormalize_franka_joints(target_normalized),
        )
        return (
            collision_loss(
                input_pc,
                cuboid_centers,
                cuboid_dims,
                cuboid_quaternions,
                cylinder_centers,
                cylinder_radii,
                cylinder_heights,
                cylinder_quaternions,
            ),
            point_match_loss(input_pc, target_pc),
        )


def compute_pose_loss_rotmat(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes position and rotation loss between predicted and target end-effector poses.

    Args:
        pred_pose (torch.Tensor): Predicted pose (B, 4, 4)
        target_pose (torch.Tensor): Target pose (B, 12): [x, y, z, flattened 3x3 rotation matrix (row-major)]

    Returns:
        position_loss (torch.Tensor): (B,) squared Euclidean position loss
        rotation_loss (torch.Tensor): (B,) squared Frobenius norm between rotation matrices
    """
    # Extract target position and rotation matrix
    target_pos = target_pose[:, 0:3]  # (B, 3)
    target_rot = target_pose[:, 3:12].view(-1, 3, 3)  # (B, 3, 3)

    # Extract predicted rotation and translation
    pred_rot = pred_pose[:, :3, :3]  # (B, 3, 3)
    pred_pos = pred_pose[:, :3, 3]   # (B, 3)

    # Position loss (squared Euclidean distance)
    position_loss = torch.sum((pred_pos - target_pos) ** 2, dim=1)  # (B,)

    # Rotation loss (Chordal Distance = squared Frobenius norm)
    rotation_loss = torch.sum((pred_rot - target_rot) ** 2, dim=(1, 2))  # (B,)

    return position_loss, rotation_loss


import torch
import time
import numpy as np
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRealRobot
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Union
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
# Assuming loss.py contains compute_pose_loss_rotmat and collision_loss
from loss import compute_pose_loss_rotmat, collision_loss


import torch
import time
import numpy as np
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRealRobot
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Union
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
# Assuming loss.py contains compute_pose_loss_rotmat and collision_loss
from loss import compute_pose_loss_rotmat, collision_loss


# A helper constant to define which link pairs to check for self-collision.
# This avoids checking adjacent links, which are always "in collision".
# A more robust solution could generate this automatically from the URDF.
FRANKA_SELF_COLLISION_PAIRS = [
    ('panda_link0', 'panda_link2'), ('panda_link0', 'panda_link3'),
    ('panda_link0', 'panda_link4'), ('panda_link0', 'panda_link5'),
    ('panda_link0', 'panda_link6'), ('panda_link0', 'panda_link7'),
    ('panda_link0', 'panda_hand'),
    ('panda_link1', 'panda_link3'), ('panda_link1', 'panda_link4'),
    ('panda_link1', 'panda_link5'), ('panda_link1', 'panda_link6'),
    ('panda_link1', 'panda_link7'), ('panda_link1', 'panda_hand'),
    ('panda_link2', 'panda_link4'), ('panda_link2', 'panda_link5'),
    ('panda_link2', 'panda_link6'), ('panda_link2', 'panda_link7'),
    ('panda_link2', 'panda_hand'),
    ('panda_link3', 'panda_link5'), ('panda_link3', 'panda_link6'),
    ('panda_link3', 'panda_link7'), ('panda_link3', 'panda_hand'),
    ('panda_link4', 'panda_link6'), ('panda_link4', 'panda_link7'),
    ('panda_link4', 'panda_hand'),
    ('panda_link5', 'panda_hand'),
    # Note: 'panda_hand' in the robofin URDF includes the gripper fingers.
]


def trajectory_opt_pointcld_self_collision(
    trajectory_init: np.ndarray,
    target_pose: SE3,
    obstacle_points: np.ndarray,
    gpu_fk_sampler: FrankaSampler,
    num_iterations: int = 20,
    learning_rate: float = 1e-4,
    goal_weight: float = 0.1,
    position_weight: float = 5.0,
    orientation_weight: float = 0.1,
    smoothness_weight: float = 1.0,
    collision_weight: float = 20.0,
    self_collision_weight: float = 80.0,  # New: Weight for self-collision
    collision_threshold: float = 0.03,
    num_robot_points: int = 512,
    freeze_first_config: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Optimizes a robot trajectory using gradient descent, including a
    self-collision penalty based on per-link point clouds.

    Args:
        trajectory_init (np.ndarray): The initial trajectory (T, 7).
        target_pose (SE3): The target end-effector pose.
        obstacle_points (np.ndarray): Point cloud of external obstacles (N, 3).
        gpu_fk_sampler (FrankaSampler): A differentiable FK and point cloud sampler.
        num_iterations (int): Number of optimization iterations.
        learning_rate (float): Learning rate for the Adam optimizer.
        goal_weight (float): Overall weight for the goal-reaching loss.
        position_weight (float): Weight for the position component of the goal loss.
        orientation_weight (float): Weight for the orientation component of the goal loss.
        smoothness_weight (float): Weight for the trajectory smoothness penalty.
        collision_weight (float): Weight for the environment collision penalty.
        self_collision_weight (float): Weight for the self-collision penalty.
        collision_threshold (float): Distance threshold (in meters) for collision penalties.
        num_robot_points (int): Number of points to sample on the robot surface.
        freeze_first_config (bool): If True, the first configuration of the trajectory is not optimized.
        verbose (bool): If True, prints loss values during optimization.

    Returns:
        np.ndarray: The optimized trajectory (T, 7).
    """
    # Convert to PyTorch tensor with gradient tracking
    trajectory = torch.tensor(
        trajectory_init, dtype=torch.float32, device="cuda", requires_grad=True
    )

    # Define joint limits for clamping
    joint_limits = FrankaRealRobot.JOINT_LIMITS
    lower_limits = torch.tensor(joint_limits[:, 0], dtype=torch.float32, device="cuda")
    upper_limits = torch.tensor(joint_limits[:, 1], dtype=torch.float32, device="cuda")

    # Prepare target pose and obstacle points for loss calculation
    with torch.no_grad():
        target_position = torch.as_tensor(
            target_pose.matrix[:3, 3], dtype=torch.float32, device="cuda"
        )
        target_rot_mat = torch.as_tensor(
            target_pose.matrix[:3, :3], dtype=torch.float32, device="cuda"
        ).flatten()
        target_pose_input = torch.cat((target_position, target_rot_mat)).unsqueeze(0)

        if obstacle_points.size > 0:
            obstacle_tensor = torch.tensor(
                obstacle_points, dtype=torch.float32, device="cuda"
            )
            expanded_obstacle = obstacle_tensor.unsqueeze(0).expand(
                len(trajectory), -1, -1
            )
            has_obstacles = True
        else:
            has_obstacles = False

    # Setup optimizer
    optimizer = torch.optim.Adam([trajectory], lr=learning_rate, weight_decay=1e-4)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # 1. Goal-reaching loss (position and orientation)
        final_pose = gpu_fk_sampler.end_effector_pose(trajectory[-1:])
        pos_loss, rot_loss = compute_pose_loss_rotmat(final_pose, target_pose_input)
        goal_loss = position_weight * pos_loss + orientation_weight * rot_loss

        # 2. Smoothness loss (penalizes acceleration)
        if len(trajectory) > 2:
            acc = trajectory[:-2] - 2 * trajectory[1:-1] + trajectory[2:]
            smooth_loss = torch.mean(torch.sum(acc**2, dim=-1))
        else:
            smooth_loss = torch.tensor(0.0, device="cuda")

        # 3. Environment Collision Loss
        if has_obstacles:
            # Sample a single point cloud for the whole robot for env collision
            input_pc = gpu_fk_sampler.sample(trajectory, num_robot_points)
            dists = torch.cdist(input_pc, expanded_obstacle)
            min_dists = torch.min(dists, dim=2)[0]
            colli_loss = torch.sum(torch.clamp(collision_threshold - min_dists, min=0))
        else:
            colli_loss = torch.tensor(0.0, device="cuda")
            
        # 4. Self-Collision Loss
        # Sample point clouds per link. The sampler returns a dict: {link_name: (T, K_i, 3)}
        per_link_pcs = gpu_fk_sampler.sample_per_link(trajectory, total_points=num_robot_points)

        self_colli_loss = torch.tensor(0.0, device="cuda")
        for link_a_name, link_b_name in FRANKA_SELF_COLLISION_PAIRS:
            if link_a_name in per_link_pcs and link_b_name in per_link_pcs:
                pc_a = per_link_pcs[link_a_name]  # Shape: (T, K_a, 3)
                pc_b = per_link_pcs[link_b_name]  # Shape: (T, K_b, 3)

                # Batched distance computation across all trajectory timesteps
                dists = torch.cdist(pc_a, pc_b)  # Shape: (T, K_a, K_b) 

                # Find the minimum distance for each timestep by flattening point pairs
                min_dists_per_timestep = torch.min(dists.view(dists.shape[0], -1), dim=1)[0] # Shape: (T,)

                # Hinge loss for this pair, aggregated over the trajectory
                pair_loss = torch.sum(torch.clamp(collision_threshold - min_dists_per_timestep, min=0))
                self_colli_loss += pair_loss
        
        # check if self_colli_loss is 0
        if self_colli_loss != 0:
            print(f"Self-collision loss: {self_colli_loss.item():.8f} for pair {link_a_name} and {link_b_name}")

        # 5. Combined loss
        total_loss = (
            goal_weight * goal_loss +
            smoothness_weight * smooth_loss +
            collision_weight * colli_loss +
            self_collision_weight * self_colli_loss
        )

        # Backpropagation
        total_loss.backward()

        # Zero out gradient for the first configuration if it's meant to be fixed
        if freeze_first_config and trajectory.grad is not None:
            with torch.no_grad():
                trajectory.grad[0].zero_()

        # Optimizer step
        optimizer.step()

        # Clamp the trajectory to be within the robot's joint limits
        with torch.no_grad():
            trajectory.data = torch.max(
                torch.min(trajectory.data, upper_limits), lower_limits
            )

        # Progress reporting
        if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
            print(f"Iter {iteration}: "
                  f"Pos={pos_loss.item():.4f}, "
                  f"Rot={rot_loss.item():.4f}, "
                  f"Smooth={smooth_loss.item():.4f}, "
                  f"Colli={colli_loss.item():.4f}, "
                  f"SelfColli={self_colli_loss.item():.4f}, " # Added self-collision to printout
                  f"Total={total_loss.item():.4f}")

    return trajectory.detach().cpu().numpy()


def trajectory_opt_pointcld(
    trajectory_init: np.ndarray,
    target_pose: SE3,
    obstacle_points: np.ndarray,
    gpu_fk_sampler: FrankaSampler,
    num_iterations: int = 30,
    learning_rate: float = 1e-3,
    goal_weight: float = 100,
    position_weight: float = 5.0,  # New: separate weight for position
    orientation_weight: float = 0.1,  # New: separate weight for orientation
    smoothness_weight: float = 1,
    collision_weight: float = 20,
    collision_threshold: float = 0.03,  # New: configurable collision threshold (3cm)
    num_robot_points: int = 512,
    freeze_first_config: bool = True,  # New: option to freeze initial config
    verbose: bool = True,  # New: verbose output for debugging
) -> np.ndarray:
    """
    Optimizes a robot trajectory using gradient descent with enhanced features:
    - Separate weights for position and orientation
    - Configurable collision threshold
    - Option to freeze initial configuration
    - Improved timing and debugging
    """
    # Timing diagnostics
    timers = {
        'total': 0,
        'setup': 0,
        'goal_loss': 0,
        'smooth_loss': 0,
        'collision_loss': 0,
        'backward': 0,
        'step': 0,
        'fk': 0
    }

    timers['total'] = time.time()

    # Convert to PyTorch tensor with gradient tracking
    trajectory = torch.tensor(
        trajectory_init, dtype=torch.float32, device="cuda", requires_grad=True
    )

    # +++ START: JOINT LIMITS MODIFICATION +++
    # Define joint limits as PyTorch tensors on the correct device
    joint_limits = FrankaRealRobot.JOINT_LIMITS
    lower_limits = torch.tensor(joint_limits[:, 0], dtype=torch.float32, device="cuda")
    upper_limits = torch.tensor(joint_limits[:, 1], dtype=torch.float32, device="cuda")
    # +++ END: JOINT LIMITS MODIFICATION +++

    # Prepare target pose and obstacle points
    setup_start = time.time()
    with torch.no_grad():
        # Target pose setup
        target_position = torch.as_tensor(
            target_pose.matrix[:3, 3], dtype=torch.float32, device="cuda"
        )
        target_rot_mat = torch.as_tensor(
            target_pose.matrix[:3, :3], dtype=torch.float32, device="cuda"
        ).flatten()
        target_pose_input = torch.cat((target_position, target_rot_mat)).unsqueeze(0)

        # Prepare obstacle point cloud
        if obstacle_points.size > 0:
            obstacle_tensor = torch.tensor(
                obstacle_points, dtype=torch.float32, device="cuda"
            )
            # Expand to [T, N, 3] once (memory efficient)
            expanded_obstacle = obstacle_tensor.unsqueeze(0).expand(
                len(trajectory), -1, -1
            )
            has_obstacles = True
        else:
            expanded_obstacle = torch.zeros(
                (len(trajectory), 1, 3), dtype=torch.float32, device="cuda"
            )
            has_obstacles = False

    timers['setup'] = time.time() - setup_start

    # Setup optimizer with weight decay for better smoothness
    optimizer = torch.optim.Adam([trajectory], lr=learning_rate)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # 1. Goal-reaching loss (with separate position/orientation weights)
        goal_start = time.time()
        fk_start = time.time()
        final_pose = gpu_fk_sampler.end_effector_pose(trajectory[-1:])
        timers['fk'] += time.time() - fk_start

        pos_loss, rot_loss = compute_pose_loss_rotmat(final_pose, target_pose_input)
        goal_loss = position_weight * pos_loss + orientation_weight * rot_loss
        timers['goal_loss'] += time.time() - goal_start

        # 2. Smoothness loss (acceleration penalty)
        smooth_start = time.time()
        if len(trajectory) > 2:
            # Finite difference acceleration
            acc = trajectory[:-2] - 2 * trajectory[1:-1] + trajectory[2:]
            smooth_loss = torch.mean(torch.sum(acc**2, dim=-1))
        else:
            smooth_loss = torch.tensor(0.0, device="cuda")
        timers['smooth_loss'] += time.time() - smooth_start

        # 3. Point cloud collision loss
        colli_start = time.time()
        input_pc = gpu_fk_sampler.sample(trajectory, num_robot_points)

        if has_obstacles:
            # Vectorized distance computation
            dists = torch.cdist(input_pc, expanded_obstacle)  # [T, M, N]
            min_dists = torch.min(dists, dim=2)[0]  # [T, M]

            # Hinge loss with configurable threshold
            colli_loss = torch.sum(torch.clamp(collision_threshold - min_dists, min=0))
        else:
            colli_loss = torch.tensor(0.0, device="cuda")
        timers['collision_loss'] += time.time() - colli_start

        # Combined loss
        total_loss = (
            goal_weight * goal_loss +
            smoothness_weight * smooth_loss +
            collision_weight * colli_loss
        )

        # Backpropagation
        backward_start = time.time()
        total_loss.backward()
        timers['backward'] += time.time() - backward_start

        # Zero out gradient for first configuration if needed
        if freeze_first_config and trajectory.grad is not None:
            with torch.no_grad():
                trajectory.grad[0].zero_()

        # Optimizer step
        step_start = time.time()
        optimizer.step()
        timers['step'] += time.time() - step_start

        # +++ START: JOINT LIMITS MODIFICATION +++
        # Clamp the trajectory to be within the joint limits
        with torch.no_grad():
            trajectory.data = torch.max(
                torch.min(trajectory.data, upper_limits), lower_limits
            )
        # +++ END: JOINT LIMITS MODIFICATION +++

        # Progress reporting
        if verbose and (iteration % 10 == 0 or iteration == num_iterations-1):
            print(f"Iter {iteration}: "
                  f"Pos={pos_loss.item():.4f}, "
                  f"Rot={rot_loss.item():.4f}, "
                  f"Smooth={smooth_loss.item():.4f}, "
                  f"Colli={colli_loss.item():.4f}, "
                  f"Total={total_loss.item():.4f}")

    timers['total'] = time.time() - timers['total']

    if verbose:
        print("\n=== Timing Breakdown ===")
        for name, duration in timers.items():
            print(f"{name:15s}: {duration:.4f}s ({duration/timers['total']*100:.1f}%)")

        print("\n=== Loss Weights ===")
        print(f"Position: {position_weight}, Orientation: {orientation_weight}")
        print(f"Goal: {goal_weight}, Smooth: {smoothness_weight}, Colli: {collision_weight}")
        print(f"Collision Threshold: {collision_threshold}m")

    return trajectory.detach().cpu().numpy()

