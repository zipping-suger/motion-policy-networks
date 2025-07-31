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
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Union
from geometrout.primitive import Cuboid, Cylinder
from geometrout.transform import SE3
from loss import compute_pose_loss_rotmat, collision_loss

def trajectory_opt_pointcld(
    trajectory_init: np.ndarray,
    target_pose: SE3,
    obstacle_points: np.ndarray,  # [N, 3] numpy array of obstacle points
    gpu_fk_sampler: FrankaSampler,
    num_iterations: int = 35,
    learning_rate: float = 1e-1,
    goal_weight: float = 0.1,
    smoothness_weight: float = 1,
    collision_weight: float = 10,
    num_robot_points: int = 512,
) -> np.ndarray:
    """
    Optimizes a robot trajectory using gradient descent to minimize:
    - Goal-reaching error (position + orientation)
    - Trajectory smoothness (acceleration penalty)
    - Collision with obstacles (using point cloud distance)
    """
    import time
    total_time = 0
    setup_time = 0
    goal_loss_time = 0
    smooth_loss_time = 0
    colli_loss_time = 0
    backward_time = 0
    step_time = 0

    total_start = time.time()

    # Convert to PyTorch tensor with gradient tracking
    trajectory = torch.tensor(
        trajectory_init, dtype=torch.float32, device="cuda", requires_grad=True
    )

    # Prepare target pose and obstacle points
    setup_start = time.time()
    with torch.no_grad():
        # Target pose setup (same as primitive version)
        target_position = torch.as_tensor(
            target_pose.matrix[:3, 3], dtype=torch.float32, device="cuda"
        )
        target_rot_mat = torch.as_tensor(
            target_pose.matrix[:3, :3].flatten(), dtype=torch.float32, device="cuda"
        )
        target_pose_input = torch.cat((target_position, target_rot_mat)).unsqueeze(0)
        
        # Prepare obstacle point cloud
        if obstacle_points.size > 0:
            obstacle_tensor = torch.tensor(
                obstacle_points, dtype=torch.float32, device="cuda"
            )
            # Expand to batch dimension [T, N, 3]
            expanded_obstacle = obstacle_tensor.unsqueeze(0).repeat(
                len(trajectory), 1, 1
            )
            print(f"Obstacle points shape: {expanded_obstacle.shape}")
            has_obstacles = True
        else:
            # Create dummy obstacle points to avoid runtime errors
            expanded_obstacle = torch.zeros(
                (len(trajectory), 1, 3), dtype=torch.float32, device="cuda"
            )
            has_obstacles = False

    setup_time = time.time() - setup_start

    # Setup optimizer
    optimizer = torch.optim.Adam([trajectory], lr=learning_rate)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # 1. Goal-reaching loss
        goal_start = time.time()
        final_pose = gpu_fk_sampler.end_effector_pose(trajectory[-1:])
        pos_loss, rot_loss = compute_pose_loss_rotmat(final_pose, target_pose_input)
        goal_loss = pos_loss + 0.1 * rot_loss
        goal_loss_time += time.time() - goal_start

        # 2. Smoothness loss
        smooth_start = time.time()
        if len(trajectory) > 2:
            acc = trajectory[:-2] - 2 * trajectory[1:-1] + trajectory[2:]
            smooth_loss = torch.mean(torch.sum(acc**2, dim=-1))
        else:
            smooth_loss = torch.sum(trajectory * 0.0)
        smooth_loss_time += time.time() - smooth_start

        # 3. Point cloud collision loss
        colli_start = time.time()
        input_pc = gpu_fk_sampler.sample(trajectory, num_robot_points)
        
        if has_obstacles:
            # Calculate distance to nearest obstacle point
            dists = torch.cdist(input_pc, expanded_obstacle)  # [T, M, N]
            min_dists, _ = torch.min(dists, dim=2)  # [T, M]
            
            # Hinge loss: penalize points closer than 3cm to obstacles
            colli_loss = torch.mean(torch.clamp(0.03 - min_dists, min=0))
        else:
            colli_loss = torch.tensor(0.0, device="cuda")
        
        colli_loss_time += time.time() - colli_start

        # Combined loss
        total_loss = (
            goal_weight * goal_loss
            + smoothness_weight * smooth_loss
            + collision_weight * colli_loss
        )

        # Backpropagation
        backward_start = time.time()
        total_loss.backward()
        backward_time += time.time() - backward_start

        # Zero out gradient for first configuration
        with torch.no_grad():
            if trajectory.grad is not None:
                trajectory.grad[0].zero_()

        # Optimizer step
        step_start = time.time()
        optimizer.step()
        step_time += time.time() - step_start

        # Progress reporting
        if iteration % 10 == 0:
            print(
                f"Iter {iteration}: "
                f"Goal={goal_loss.item():.4f}, "
                f"Smooth={smooth_loss.item():.4f}, "
                f"Colli={colli_loss.item():.4f}, "
                f"Total={total_loss.item():.4f}"
            )

    total_time = time.time() - total_start

    # Print timing results
    print("\nTiming Results:")
    print(f"Total time: {total_time:.4f}s")
    print(f"Setup time: {setup_time:.4f}s ({setup_time/total_time*100:.1f}%)")
    print(f"Goal loss time: {goal_loss_time:.4f}s ({goal_loss_time/total_time*100:.1f}%)")
    print(f"Smooth loss time: {smooth_loss_time:.4f}s ({smooth_loss_time/total_time*100:.1f}%)")
    print(f"Collision loss time: {colli_loss_time:.4f}s ({colli_loss_time/total_time*100:.1f}%)")
    print(f"Backward time: {backward_time:.4f}s ({backward_time/total_time*100:.1f}%)")
    print(f"Step time: {step_time:.4f}s ({step_time/total_time*100:.1f}%)")

    return trajectory.detach().cpu().numpy()


def trajectory_opt_primitive(
    trajectory_init: np.ndarray,
    target_pose: SE3,
    obstacles: List[Union[Cuboid, Cylinder]],
    gpu_fk_sampler: FrankaSampler,
    num_iterations: int = 35,
    learning_rate: float = 1e-1,
    goal_weight: float = 0.1,
    smoothness_weight: float = 1,
    collision_weight: float = 10,
    num_robot_points: int = 1024,
) -> np.ndarray:
    """
    Optimizes a robot trajectory using gradient descent to minimize:
    - Goal-reaching error (position + orientation)
    - Trajectory smoothness (acceleration penalty)
    - Collision with obstacles
    """
    # Initialize timers
    total_time = 0
    setup_time = 0
    goal_loss_time = 0
    smooth_loss_time = 0
    colli_loss_time = 0
    backward_time = 0
    step_time = 0

    total_start = time.time()

    # Convert to PyTorch tensor with gradient tracking
    trajectory = torch.tensor(
        trajectory_init, dtype=torch.float32, device="cuda", requires_grad=True
    )

    # Prepare target pose (no gradients needed)
    setup_start = time.time()
    with torch.no_grad():
        target_position = torch.as_tensor(
            target_pose.matrix[:3, 3], dtype=torch.float32, device="cuda"
        )
        target_rot_mat = torch.as_tensor(
            target_pose.matrix[:3, :3].flatten(), dtype=torch.float32, device="cuda"
        )
        target_pose_input = torch.cat((target_position, target_rot_mat)).unsqueeze(0)

        # Prepare obstacle tensors
        cuboids = [o for o in obstacles if isinstance(o, Cuboid)]
        cylinders = [o for o in obstacles if isinstance(o, Cylinder)]

        # Cuboid tensors
        cuboid_centers = (
            torch.tensor(
                [o.center for o in cuboids], dtype=torch.float32, device="cuda"
            )
            .unsqueeze(0)
            .repeat(len(trajectory), 1, 1)
            if cuboids
            else torch.empty(
                (len(trajectory), 0, 3), dtype=torch.float32, device="cuda"
            )
        )
        cuboid_dims = (
            torch.tensor([o.dims for o in cuboids], dtype=torch.float32, device="cuda")
            .unsqueeze(0)
            .repeat(len(trajectory), 1, 1)
            if cuboids
            else torch.empty(
                (len(trajectory), 0, 3), dtype=torch.float32, device="cuda"
            )
        )
        cuboid_quats = (
            torch.tensor(
                [o.pose.so3.wxyz for o in cuboids],
                dtype=torch.float32,
                device="cuda",
            )
            .unsqueeze(0)
            .repeat(len(trajectory), 1, 1)
            if cuboids
            else torch.empty(
                (len(trajectory), 0, 4), dtype=torch.float32, device="cuda"
            )
        )

        # Cylinder tensors
        if cylinders:
            cylinder_centers = (
                torch.tensor(
                    [o.center for o in cylinders], dtype=torch.float32, device="cuda"
                )
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )
            cylinder_radii = (
                torch.tensor(
                    [[o.radius] for o in cylinders], dtype=torch.float32, device="cuda"
                )
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )
            cylinder_heights = (
                torch.tensor(
                    [[o.height] for o in cylinders], dtype=torch.float32, device="cuda"
                )
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )
            cylinder_quats = (
                torch.tensor(
                    [o.pose.so3.wxyz for o in cylinders],
                    dtype=torch.float32,
                    device="cuda",
                )
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )
        else:
            # Dummy cylinder values
            cylinder_radii_np = np.array([[0.0]])
            cylinder_heights_np = np.array([[0.0]])
            cylinder_centers_np = np.array([[0.0, 0.0, 0.0]])
            cylinder_quats_np = np.array(
                [[1.0, 0.0, 0.0, 0.0]]
            )

            cylinder_centers = (
                torch.tensor(cylinder_centers_np, dtype=torch.float32, device="cuda")
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )
            cylinder_radii = (
                torch.tensor(cylinder_radii_np, dtype=torch.float32, device="cuda")
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )
            cylinder_heights = (
                torch.tensor(cylinder_heights_np, dtype=torch.float32, device="cuda")
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )
            cylinder_quats = (
                torch.tensor(cylinder_quats_np, dtype=torch.float32, device="cuda")
                .unsqueeze(0)
                .repeat(len(trajectory), 1, 1)
            )

        has_any_obstacles_for_collision = bool(cuboids) or bool(cylinders)

    setup_time = time.time() - setup_start

    # Setup optimizer
    optimizer = torch.optim.Adam([trajectory], lr=learning_rate)

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # 1. Goal-reaching loss
        goal_start = time.time()
        final_pose = gpu_fk_sampler.end_effector_pose(trajectory[-1:])
        pos_loss, rot_loss = compute_pose_loss_rotmat(final_pose, target_pose_input)
        goal_loss = pos_loss + 0.1 * rot_loss
        goal_loss_time += time.time() - goal_start

        # 2. Smoothness loss
        smooth_start = time.time()
        if len(trajectory) > 2:
            acc = trajectory[:-2] - 2 * trajectory[1:-1] + trajectory[2:]
            smooth_loss = torch.mean(torch.sum(acc**2, dim=-1))
        else:
            smooth_loss = torch.sum(trajectory * 0.0)
        smooth_loss_time += time.time() - smooth_start

        # 3. Collision loss
        colli_start = time.time()
        input_pc = gpu_fk_sampler.sample(trajectory, num_robot_points)
        colli_loss = collision_loss(
            input_pc,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
        )
        colli_loss_time += time.time() - colli_start

        # Combined loss
        total_loss = (
            goal_weight * goal_loss
            + smoothness_weight * smooth_loss
            + collision_weight * colli_loss
        )

        # Backpropagation
        backward_start = time.time()
        total_loss.backward()
        backward_time += time.time() - backward_start

        ### FIX START ###
        # Zero out the gradient for the first configuration to keep it fixed
        with torch.no_grad():
            if trajectory.grad is not None:
                trajectory.grad[0].zero_()
        ### FIX END ###

        # Optimizer step
        step_start = time.time()
        optimizer.step()
        step_time += time.time() - step_start

        # Progress reporting
        if iteration % 10 == 0:
            print(
                f"Iter {iteration}: "
                f"Goal={goal_loss.item():.4f}, "
                f"Smooth={smooth_loss.item():.4f}, "
                f"Colli={colli_loss.item():.4f}, "
                f"Total={total_loss.item():.4f}"
            )

    total_time = time.time() - total_start

    # Print timing results
    print("\nTiming Results:")
    print(f"Total time: {total_time:.4f}s")
    print(f"Setup time: {setup_time:.4f}s ({setup_time/total_time*100:.1f}%)")
    print(f"Goal loss time: {goal_loss_time:.4f}s ({goal_loss_time/total_time*100:.1f}%)")
    print(f"Smooth loss time: {smooth_loss_time:.4f}s ({smooth_loss_time/total_time*100:.1f}%)")
    print(f"Collision loss time: {colli_loss_time:.4f}s ({colli_loss_time/total_time*100:.1f}%)")
    print(f"Backward time: {backward_time:.4f}s ({backward_time/total_time*100:.1f}%)")
    print(f"Step time: {step_time:.4f}s ({step_time/total_time*100:.1f}%)")

    return trajectory.detach().cpu().numpy()
