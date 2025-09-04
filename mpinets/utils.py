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

from typing import Union, Tuple

import numpy as np
import torch
from robofin.robots import FrankaRobot, FrankaRealRobot

import trimesh
from robofin.torch_urdf import TorchURDF
import logging


# Overwirte the FrankaSampler from robofin to add a method that samples points per link

def transform_pointcloud(pc, transformation_matrix, in_place=True):
    """

    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography

    """
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)


class FrankaSampler:
    """
    This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running FK on a subsample
    of the per-link pointclouds that are established at initialization.

    """

    def __init__(
        self,
        device,
        num_fixed_points=None,
        use_cache=False,
        default_prismatic_value=0.025,
        with_base_link=True,
        attached_primitive=None
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self.default_prismatic_value = default_prismatic_value
        self.with_base_link = with_base_link
        self.attached_primitive = attached_primitive
        self._init_internal_(device, use_cache)

    def _init_internal_(self, device, use_cache):
        self.robot = TorchURDF.load(
            FrankaRobot.urdf, lazy_load_meshes=True, device=device
        )
        self.links = [l for l in self.robot.links if len(l.visuals)]
        if use_cache and self._init_from_cache_(device):
            return

        meshes = [
            trimesh.load(
                Path(FrankaRobot.urdf).parent / l.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for l in self.links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        if self.num_fixed_points is not None:
            num_points = np.round(
                self.num_fixed_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += self.num_fixed_points - np.sum(num_points)
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(4096 * np.array(areas) / np.sum(areas))
        self.points = {}
        for ii in range(len(meshes)):
            pc = trimesh.sample.sample_surface(meshes[ii], int(num_points[ii]))[0]
            self.points[self.links[ii].name] = torch.as_tensor(
                pc, device=device
            ).unsqueeze(0)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        if use_cache:
            points_to_save = {
                k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.points.items()
            }
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _get_cache_file_name_(self):
        if self.num_fixed_points is not None:
            return (
                FrankaRobot.pointcloud_cache
                / f"fixed_point_cloud_{self.num_fixed_points}.npy"
            )
        else:
            return FrankaRobot.pointcloud_cache / "full_point_cloud.npy"

    def _init_from_cache_(self, device):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in points.item().items()
        }
        return True

    def end_effector_pose(self, config, frame="right_gripper"):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        return fk[frame]

    def sample_end_effector(self, poses, num_points, frame="right_gripper"):
        """
        An internal method--separated so that the public facing method can
        choose whether or not to have gradients
        """
        assert poses.ndim in [2, 3]
        assert frame == "right_gripper", "Other frames not yet suppported"
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        default_cfg = torch.zeros((1, 9), device=poses.device)
        default_cfg[0, 7:] = self.default_prismatic_value
        fk = self.robot.visual_geometry_fk_batch(default_cfg)
        eff_link_names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]

        # This logic could break--really need a way to make sure that the
        # ordering is correct
        values = [
            list(fk.values())[idx]
            for idx, l in enumerate(self.links)
            if l.name in eff_link_names
        ]
        end_effector_links = [l for l in self.links if l.name in eff_link_names]
        assert len(end_effector_links) == len(values)
        fk_transforms = {}
        fk_points = []
        gripper_T_hand = torch.as_tensor(
            FrankaRobot.EFF_T_LIST[("panda_hand", "right_gripper")].inverse.matrix
        ).type_as(poses)
        # Could just invert the matrix, but matrix inversion is not implemented for half-types
        inverse_hand_transform = torch.zeros_like(values[0])
        inverse_hand_transform[:, -1, -1] = 1
        inverse_hand_transform[:, :3, :3] = values[0][:, :3, :3].transpose(1, 2)
        inverse_hand_transform[:, :3, -1] = -torch.matmul(
            inverse_hand_transform[:, :3, :3], values[0][:, :3, -1].unsqueeze(-1)
        ).squeeze(-1)
        right_gripper_transform = gripper_T_hand.unsqueeze(0) @ inverse_hand_transform
        for idx, l in enumerate(end_effector_links):
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud(
                self.points[l.name].type_as(poses),
                (right_gripper_transform @ fk_transforms[l.name]),
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        pc = transform_pointcloud(pc.repeat(poses.size(0), 1, 1), poses)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def sample(self, config, num_points=None):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints.
            For example, if using the Franka, M is 9
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None)
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name == "panda_link0" and not self.with_base_link:
                continue
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud(
                self.points[l.name]
                .float()
                .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        
        # Add attached primitive points if specified
        if self.attached_primitive is not None:
            ee_pose = self.end_effector_pose(config, frame="right_gripper")
            primitive_points = self._sample_attached_primitive(ee_pose)
            pc = torch.cat([pc, primitive_points], dim=1)
        
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def _sample_attached_primitive(self, ee_poses):
        """Sample points from the attached primitive with offset from end-effector"""
        # Get offset configuration
        offset = self.attached_primitive.get('offset', [0, 0, 0])
        offset_quat = self.attached_primitive.get('offset_quaternion', [1, 0, 0, 0])  # w, x, y, z
        
        # Convert offset to tensor
        offset_tensor = torch.tensor(offset, device=ee_poses.device, dtype=ee_poses.dtype)
        offset_quat_tensor = torch.tensor(offset_quat, device=ee_poses.device, dtype=ee_poses.dtype)
        
        # Create offset transformation matrix
        if ee_poses.dim() == 3:  # Batch of poses
            batch_size = ee_poses.shape[0]
            offset_transform = torch.eye(4, device=ee_poses.device, dtype=ee_poses.dtype)
            offset_transform = offset_transform.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Apply translation offset
            offset_transform[:, :3, 3] = offset_tensor
            
            # Apply rotation offset (convert quaternion to rotation matrix)
            w, x, y, z = offset_quat_tensor
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            
            rotation_matrix = torch.stack([
                1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy),
                2*(xy + wz),     1 - 2*(xx + zz),     2*(yz - wx),
                2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
            ], dim=0).reshape(3, 3)
            
            offset_transform[:, :3, :3] = rotation_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Combine offset with end-effector pose
            combined_poses = torch.matmul(ee_poses, offset_transform)
        else:  # Single pose
            offset_transform = torch.eye(4, device=ee_poses.device, dtype=ee_poses.dtype)
            offset_transform[:3, 3] = offset_tensor
            
            # Apply rotation offset
            w, x, y, z = offset_quat_tensor
            xx, yy, zz = x*x, y*y, z*z
            xy, xz, yz = x*y, x*z, y*z
            wx, wy, wz = w*x, w*y, w*z
            
            rotation_matrix = torch.tensor([
                [1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)],
                [2*(xy + wz),     1 - 2*(xx + zz),     2*(yz - wx)],
                [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
            ], device=ee_poses.device, dtype=ee_poses.dtype)
            
            offset_transform[:3, :3] = rotation_matrix
            
            # Combine offset with end-effector pose
            combined_poses = torch.matmul(ee_poses, offset_transform)
        
        # Now generate points based on primitive type
        if self.attached_primitive['type'] == 'cuboid':
            # Sample points on cuboid surface
            dims = torch.tensor(self.attached_primitive['dims'], 
                            device=ee_poses.device, 
                            dtype=ee_poses.dtype)
            num_points = self.attached_primitive.get('num_points', 500)
            
            # Generate points on surface of cuboid (local frame)
            points = []
            for _ in range(num_points):
                face = torch.randint(0, 6, (1,)).item()
                if face == 0:  # +x face
                    x = dims[0]/2
                    y = torch.rand(1).item() * dims[1] - dims[1]/2
                    z = torch.rand(1).item() * dims[2] - dims[2]/2
                elif face == 1:  # -x face
                    x = -dims[0]/2
                    y = torch.rand(1).item() * dims[1] - dims[1]/2
                    z = torch.rand(1).item() * dims[2] - dims[2]/2
                elif face == 2:  # +y face
                    x = torch.rand(1).item() * dims[0] - dims[0]/2
                    y = dims[1]/2
                    z = torch.rand(1).item() * dims[2] - dims[2]/2
                elif face == 3:  # -y face
                    x = torch.rand(1).item() * dims[0] - dims[0]/2
                    y = -dims[1]/2
                    z = torch.rand(1).item() * dims[2] - dims[2]/2
                elif face == 4:  # +z face
                    x = torch.rand(1).item() * dims[0] - dims[0]/2
                    y = torch.rand(1).item() * dims[1] - dims[1]/2
                    z = dims[2]/2
                else:  # -z face
                    x = torch.rand(1).item() * dims[0] - dims[0]/2
                    y = torch.rand(1).item() * dims[1] - dims[1]/2
                    z = -dims[2]/2
                points.append([x.item(), y.item(), z.item()])
            
            points = torch.tensor(points, device=ee_poses.device, dtype=ee_poses.dtype)
            
            # Add batch dimension and repeat for each pose
            if ee_poses.dim() == 3:  # Batch of poses
                points = points.unsqueeze(0).repeat(ee_poses.shape[0], 1, 1)
            else:  # Single pose
                points = points.unsqueeze(0)
            
            # Transform points to world frame using the combined pose (EE + offset)
            return transform_pointcloud(points, combined_poses, in_place=False)
        
        elif self.attached_primitive['type'] == 'cylinder':
            # Sample points on cylinder surface
            radius = self.attached_primitive['radius']
            height = self.attached_primitive['height']
            num_points = self.attached_primitive.get('num_points', 500)
            
            # Generate points on surface of cylinder (local frame)
            points = []
            for _ in range(num_points):
                # Randomly choose between side, top, or bottom
                surface_type = torch.randint(0, 3, (1,)).item()
                
                if surface_type == 0:  # Side
                    theta = torch.rand(1).item() * 2 * torch.pi
                    z = torch.rand(1).item() * height - height/2
                    x = radius * torch.cos(theta)
                    y = radius * torch.sin(theta)
                elif surface_type == 1:  # Top
                    theta = torch.rand(1).item() * 2 * torch.pi
                    r = torch.sqrt(torch.rand(1).item()) * radius
                    x = r * torch.cos(theta)
                    y = r * torch.sin(theta)
                    z = height/2
                else:  # Bottom
                    theta = torch.rand(1).item() * 2 * torch.pi
                    r = torch.sqrt(torch.rand(1).item()) * radius
                    x = r * torch.cos(theta)
                    y = r * torch.sin(theta)
                    z = -height/2
                points.append([x.item(), y.item(), z.item()])
            
            points = torch.tensor(points, device=ee_poses.device, dtype=ee_poses.dtype)
            
            # Add batch dimension and repeat for each pose
            if ee_poses.dim() == 3:  # Batch of poses
                points = points.unsqueeze(0).repeat(ee_poses.shape[0], 1, 1)
            else:  # Single pose
                points = points.unsqueeze(0)
            
            # Transform points to world frame using the combined pose (EE + offset)
            return transform_pointcloud(points, combined_poses, in_place=False)
        
        elif self.attached_primitive['type'] == 'sphere':
            # Sample points on sphere surface
            radius = self.attached_primitive['radius']
            num_points = self.attached_primitive.get('num_points', 500)
            
            # Generate points on surface of sphere (local frame)
            points = []
            for _ in range(num_points):
                # Uniform sampling on sphere surface
                theta = torch.rand(1).item() * 2 * torch.pi
                phi = torch.acos(2 * torch.rand(1).item() - 1)
                x = radius * torch.sin(phi) * torch.cos(theta)
                y = radius * torch.sin(phi) * torch.sin(theta)
                z = radius * torch.cos(phi)
                points.append([x.item(), y.item(), z.item()])
            
            points = torch.tensor(points, device=ee_poses.device, dtype=ee_poses.dtype)
            
            # Add batch dimension and repeat for each pose
            if ee_poses.dim() == 3:  # Batch of poses
                points = points.unsqueeze(0).repeat(ee_poses.shape[0], 1, 1)
            else:  # Single pose
                points = points.unsqueeze(0)
            
            # Transform points to world frame using the combined pose (EE + offset)
            return transform_pointcloud(points, combined_poses, in_place=False)
        
        else:
            raise ValueError(f"Unsupported primitive type: {self.attached_primitive['type']}")
    
    def sample_per_link(self, config, total_points=None):
        """
        Samples points from each link's surface separately, distributing points proportionally
        to each link's surface area, and returns them in a dictionary.
        
        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints. For example, if using the Franka, M is 9
        total_points : Total number of points to sample across all links (optional)
            If None, uses all pre-sampled points for each link
        
        Returns
        -------
        Dictionary where keys are link names and values are point clouds:
        - If input is (M,): returns dict of (K, 3) tensors where K is num points for that link
        - If input is (N, M): returns dict of (N, K, 3) tensors where K is num points for that link
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        
        # Get relative sizes of each link (based on initialization)
        link_sizes = {l.name: self.points[l.name].shape[1] for l in self.links 
                    if not (l.name == "panda_link0" and not self.with_base_link)}
        total_fixed_points = sum(link_sizes.values())
        
        per_link_pcs = {}
        for idx, l in enumerate(self.links):
            if l.name == "panda_link0" and not self.with_base_link:
                continue
                
            # Get all pre-sampled points for this link
            link_pc = self.points[l.name].float().repeat((values[idx].shape[0], 1, 1))
            
            # Transform points using FK
            transformed_pc = transform_pointcloud(
                link_pc,
                values[idx],
                in_place=True,
            )
            
            # Subsample proportionally if total_points is specified
            if total_points is not None:
                # Calculate how many points this link should get
                link_points = max(1, int(round(
                    total_points * (link_sizes[l.name] / total_fixed_points)
                )))
                
                if transformed_pc.shape[1] < link_points:
                    # If we don't have enough pre-sampled points, use all we have
                    link_points = transformed_pc.shape[1]
                
                if link_points < transformed_pc.shape[1]:
                    transformed_pc = transformed_pc[
                        :, 
                        np.random.choice(transformed_pc.shape[1], link_points, replace=False), 
                        :
                    ]
            
            # Squeeze if input was 1D
            if squeeze_output:
                transformed_pc = transformed_pc.squeeze(0)
                
            per_link_pcs[l.name] = transformed_pc
        
        return per_link_pcs


def _normalize_franka_joints_numpy(
    batch_trajectory: np.ndarray,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> np.ndarray:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is the numpy version

    :param batch_trajectory np.ndarray: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype np.ndarray: An array with the same dimensions as the input
    """
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = robot.JOINT_LIMITS
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.shape[0] == robot.DOF)
        or (batch_trajectory.ndim == 2 and batch_trajectory.shape[1] == robot.DOF)
        or (batch_trajectory.ndim == 3 and batch_trajectory.shape[2] == robot.DOF)
    )
    normalized = (batch_trajectory - franka_limits[:, 0]) / (
        franka_limits[:, 1] - franka_limits[:, 0]
    ) * (limits[1] - limits[0]) + limits[0]
    return normalized


def _normalize_franka_joints_torch(
    batch_trajectory: torch.Tensor,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> torch.Tensor:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is the torch version

    :param batch_trajectory torch.Tensor: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype torch.Tensor: A tensor with the same dimensions as the input
    """
    assert isinstance(batch_trajectory, torch.Tensor)
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = torch.as_tensor(robot.JOINT_LIMITS).type_as(batch_trajectory)
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.size(0) == robot.DOF)
        or (batch_trajectory.ndim == 2 and batch_trajectory.size(1) == robot.DOF)
        or (batch_trajectory.ndim == 3 and batch_trajectory.size(2) == robot.DOF)
    )
    return (batch_trajectory - franka_limits[:, 0]) / (
        franka_limits[:, 1] - franka_limits[:, 0]
    ) * (limits[1] - limits[0]) + limits[0]


def normalize_franka_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is semantic sugar that dispatches to the correct implementation.

    :param batch_trajectory Union[np.ndarray, torch.Tensor]: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype Union[np.ndarray, torch.Tensor]: A tensor or numpy array with the same dimensions
                                            and type as the input
    :raises NotImplementedError: Raises an error if another data type (e.g. a list) is passed in
    """
    if isinstance(batch_trajectory, torch.Tensor):
        return _normalize_franka_joints_torch(
            batch_trajectory, limits=limits, use_real_constraints=True
        )
    elif isinstance(batch_trajectory, np.ndarray):
        return _normalize_franka_joints_numpy(
            batch_trajectory, limits=limits, use_real_constraints=True
        )
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")


def _unnormalize_franka_joints_numpy(
    batch_trajectory: np.ndarray,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> np.ndarray:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is the numpy version and the inverse of `_normalize_franka_joints_numpy`.

    :param batch_trajectory np.ndarray: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype np.ndarray: An array with the same dimensions as the input
    """
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = robot.JOINT_LIMITS
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.shape[0] == robot.DOF)
        or (batch_trajectory.ndim == 2 and batch_trajectory.shape[1] == robot.DOF)
        or (batch_trajectory.ndim == 3 and batch_trajectory.shape[2] == robot.DOF)
    )
    assert np.all(batch_trajectory >= limits[0])
    assert np.all(batch_trajectory <= limits[1])
    franka_limit_range = franka_limits[:, 1] - franka_limits[:, 0]
    franka_lower_limit = franka_limits[:, 0]
    for _ in range(batch_trajectory.ndim - 1):
        franka_limit_range = franka_limit_range[np.newaxis, ...]
        franka_lower_limit = franka_lower_limit[np.newaxis, ...]
    unnormalized = (batch_trajectory - limits[0]) * franka_limit_range / (
        limits[1] - limits[0]
    ) + franka_lower_limit

    return unnormalized


def _unnormalize_franka_joints_torch(
    batch_trajectory: torch.Tensor,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> torch.Tensor:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is the torch version and the inverse of `_normalize_franka_joints_torch`.

    :param batch_trajectory torch.Tensor: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype torch.Tensor: A tensor with the same dimensions as the input
    """
    assert isinstance(batch_trajectory, torch.Tensor)
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = torch.as_tensor(robot.JOINT_LIMITS).type_as(batch_trajectory)
    dof = franka_limits.size(0)
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.size(0) == dof)
        or (batch_trajectory.ndim == 2 and batch_trajectory.size(1) == dof)
        or (batch_trajectory.ndim == 3 and batch_trajectory.size(2) == dof)
    )
    assert torch.all(batch_trajectory >= limits[0])
    assert torch.all(batch_trajectory <= limits[1])
    franka_limit_range = franka_limits[:, 1] - franka_limits[:, 0]
    franka_lower_limit = franka_limits[:, 0]
    for _ in range(batch_trajectory.ndim - 1):
        franka_limit_range = franka_limit_range.unsqueeze(0)
        franka_lower_limit = franka_lower_limit.unsqueeze(0)
    return (batch_trajectory - limits[0]) * franka_limit_range / (
        limits[1] - limits[0]
    ) + franka_lower_limit


def unnormalize_franka_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is semantic sugar that dispatches to the correct implementation, the inverse of
    `normalize_franka_joints`.

    :param batch_trajectory Union[np.ndarray, torch.Tensor]: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype Union[np.ndarray, torch.Tensor]: A tensor or numpy array with the same dimensions
                                            and type as the input
    :raises NotImplementedError: Raises an error if another data type (e.g. a list) is passed in
    """
    if isinstance(batch_trajectory, torch.Tensor):
        return _unnormalize_franka_joints_torch(
            batch_trajectory, limits=limits, use_real_constraints=use_real_constraints
        )
    elif isinstance(batch_trajectory, np.ndarray):
        return _unnormalize_franka_joints_numpy(
            batch_trajectory, limits=limits, use_real_constraints=use_real_constraints
        )
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")

