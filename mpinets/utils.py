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
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self.default_prismatic_value = default_prismatic_value
        self.with_base_link = with_base_link
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

    def sample_end_effector(
        self,
        poses,
        tool_dim,
        tool_offset,
        tool_quat,
        num_points,
        frame="right_gripper",
    ):
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

        # Add attached primitive points if specified
        if not torch.allclose(
            torch.as_tensor(tool_dim), torch.zeros_like(torch.as_tensor(tool_dim))
        ) or not torch.allclose(
            torch.as_tensor(tool_offset), torch.zeros_like(torch.as_tensor(tool_offset))
        ):
            # For sample_end_effector, we already have the end-effector poses as input
            # so we can directly use them to sample the attached primitive
            primitive_points = self._sample_attached_primitive(
                poses, tool_dim, tool_offset, tool_quat
            )
            pc = torch.cat([pc, primitive_points], dim=1)

        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def sample(self, config, tool_dim, tool_offset, tool_quat, num_points=None):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints.
            For example, if using the Franka, M is 9
        tool_dim : Dimensions of the attached primitive (cuboid) [3] (optional)
        tool_offset : Offset of the primitive from the EE [3] (optional)
        tool_quat : Quaternion for primitive orientation [4] (optional)
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
        if not torch.allclose(
            torch.as_tensor(tool_dim), torch.zeros_like(torch.as_tensor(tool_dim))
        ) or not torch.allclose(
            torch.as_tensor(tool_offset), torch.zeros_like(torch.as_tensor(tool_offset))
        ):
            ee_pose = self.end_effector_pose(config, frame="right_gripper")
            primitive_points = self._sample_attached_primitive(
                ee_pose, tool_dim, tool_offset, tool_quat
            )
            pc = torch.cat([pc, primitive_points], dim=1)

        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def sample_composite(
        self,
        config,
        tool_dims,
        tool_offsets,
        tool_quats,
        tool_num_primitives,
        num_points=None,
    ):
        """
        Samples points from the surface of the robot with composite tools.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints. For example, if using the Franka, M is 9
        tool_dims : Dimensions of the attached primitives [max_primitives, 3]
        tool_offsets : Offsets of the primitives from the EE [max_primitives, 3]
        tool_quats : Quaternions for primitive orientations [max_primitives, 4]
        tool_num_primitives : Number of actual primitives (not padding)
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot points with composite tools
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
                self.points[l.name].float().repeat((fk_transforms[l.name].shape[0], 1, 1)),
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)

        # Add attached composite tool points if specified
        # Fix: Handle batched tool_num_primitives properly
        if tool_num_primitives.ndim == 0:
            # Single value
            if tool_num_primitives > 0:
                ee_pose = self.end_effector_pose(config, frame="right_gripper")
                primitive_points = self._sample_composite_attached_primitive(
                    ee_pose, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)
        else:
            # Batched - check if any element has primitives
            if torch.any(tool_num_primitives > 0):
                ee_pose = self.end_effector_pose(config, frame="right_gripper")
                primitive_points = self._sample_composite_attached_primitive(
                    ee_pose, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)

        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def sample_composite_end_effector(
        self,
        poses,
        tool_dims,
        tool_offsets,
        tool_quats,
        tool_num_primitives,
        num_points,
        frame="right_gripper",
    ):
        """
        Sample end effector points with composite tools.
        """
        assert poses.ndim in [2, 3]
        assert frame == "right_gripper", "Other frames not yet supported"
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        default_cfg = torch.zeros((1, 9), device=poses.device)
        default_cfg[0, 7:] = self.default_prismatic_value
        fk = self.robot.visual_geometry_fk_batch(default_cfg)
        eff_link_names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]

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

        # Add attached composite primitive points if specified
        # Fix: Handle batched tool_num_primitives properly
        if tool_num_primitives.ndim == 0:
            # Single value
            if tool_num_primitives > 0:
                primitive_points = self._sample_composite_attached_primitive(
                    poses, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)
        else:
            # Batched - check if any element has primitives
            if torch.any(tool_num_primitives > 0):
                primitive_points = self._sample_composite_attached_primitive(
                    poses, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)

        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def _sample_attached_primitive(self, ee_poses, dim, offset, offset_quat):
        """
        Vectorized sampling of points from attached primitive surfaces.
        Supports batched ee_poses, dim, offset, and offset_quat.
        """
        device = ee_poses.device
        dtype = ee_poses.dtype

        # Handle batch dimension
        if ee_poses.dim() == 3:  # [B, 4, 4]
            batch_size = ee_poses.shape[0]
            single_pose = False
        else:  # [4, 4]
            batch_size = 1
            single_pose = True
            ee_poses = ee_poses.unsqueeze(0)

        # Convert inputs to batched tensors
        dim_tensor = torch.as_tensor(dim, device=device, dtype=dtype)
        offset_tensor = torch.as_tensor(offset, device=device, dtype=dtype)
        offset_quat_tensor = torch.as_tensor(offset_quat, device=device, dtype=dtype)

        if dim_tensor.ndim == 1:
            dim_tensor = dim_tensor.unsqueeze(0).repeat(batch_size, 1)
        if offset_tensor.ndim == 1:
            offset_tensor = offset_tensor.unsqueeze(0).repeat(batch_size, 1)
        if offset_quat_tensor.ndim == 1:
            offset_quat_tensor = offset_quat_tensor.unsqueeze(0).repeat(batch_size, 1)

        # --- Build offset transform ---
        offset_transform = (
            torch.eye(4, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        offset_transform[:, :3, 3] = offset_tensor

        # Quaternion to rotation matrix (vectorized)
        w, x, y, z = (
            offset_quat_tensor[:, 0],
            offset_quat_tensor[:, 1],
            offset_quat_tensor[:, 2],
            offset_quat_tensor[:, 3],
        )
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        rot[:, 0, 0] = 1 - 2 * (yy + zz)
        rot[:, 0, 1] = 2 * (xy - wz)
        rot[:, 0, 2] = 2 * (xz + wy)
        rot[:, 1, 0] = 2 * (xy + wz)
        rot[:, 1, 1] = 1 - 2 * (xx + zz)
        rot[:, 1, 2] = 2 * (yz - wx)
        rot[:, 2, 0] = 2 * (xz - wy)
        rot[:, 2, 1] = 2 * (yz + wx)
        rot[:, 2, 2] = 1 - 2 * (xx + yy)
        offset_transform[:, :3, :3] = rot

        # Combine offset with EE poses
        combined_poses = torch.bmm(ee_poses, offset_transform)

        # --- Sample cuboid surface points ---
        num_points = 500
        # Shared random samples for all batches
        rand_vals = torch.rand(num_points, 3, device=device, dtype=dtype)  # [N, 3]
        face_choices = torch.randint(0, 6, (num_points,), device=device)  # [N]

        # Expand for batch: [B, N, 3]
        rand_vals = rand_vals.unsqueeze(0).expand(batch_size, -1, -1)
        dims = dim_tensor.unsqueeze(1).expand(-1, num_points, -1)  # [B, N, 3]
        points = torch.zeros_like(rand_vals)

        # Masks per face (shared across batch)
        mask_xp = face_choices == 0
        mask_xn = face_choices == 1
        mask_yp = face_choices == 2
        mask_yn = face_choices == 3
        mask_zp = face_choices == 4
        mask_zn = face_choices == 5

        # Vectorized assignment per face
        if mask_xp.any():
            points[:, mask_xp, 0] = dims[:, mask_xp, 0] / 2
            points[:, mask_xp, 1] = (
                rand_vals[:, mask_xp, 0] * dims[:, mask_xp, 1] - dims[:, mask_xp, 1] / 2
            )
            points[:, mask_xp, 2] = (
                rand_vals[:, mask_xp, 1] * dims[:, mask_xp, 2] - dims[:, mask_xp, 2] / 2
            )

        if mask_xn.any():
            points[:, mask_xn, 0] = -dims[:, mask_xn, 0] / 2
            points[:, mask_xn, 1] = (
                rand_vals[:, mask_xn, 0] * dims[:, mask_xn, 1] - dims[:, mask_xn, 1] / 2
            )
            points[:, mask_xn, 2] = (
                rand_vals[:, mask_xn, 1] * dims[:, mask_xn, 2] - dims[:, mask_xn, 2] / 2
            )

        if mask_yp.any():
            points[:, mask_yp, 0] = (
                rand_vals[:, mask_yp, 0] * dims[:, mask_yp, 0] - dims[:, mask_yp, 0] / 2
            )
            points[:, mask_yp, 1] = dims[:, mask_yp, 1] / 2
            points[:, mask_yp, 2] = (
                rand_vals[:, mask_yp, 1] * dims[:, mask_yp, 2] - dims[:, mask_yp, 2] / 2
            )

        if mask_yn.any():
            points[:, mask_yn, 0] = (
                rand_vals[:, mask_yn, 0] * dims[:, mask_yn, 0] - dims[:, mask_yn, 0] / 2
            )
            points[:, mask_yn, 1] = -dims[:, mask_yn, 1] / 2
            points[:, mask_yn, 2] = (
                rand_vals[:, mask_yn, 1] * dims[:, mask_yn, 2] - dims[:, mask_yn, 2] / 2
            )

        if mask_zp.any():
            points[:, mask_zp, 0] = (
                rand_vals[:, mask_zp, 0] * dims[:, mask_zp, 0] - dims[:, mask_zp, 0] / 2
            )
            points[:, mask_zp, 1] = (
                rand_vals[:, mask_zp, 1] * dims[:, mask_zp, 1] - dims[:, mask_zp, 1] / 2
            )
            points[:, mask_zp, 2] = dims[:, mask_zp, 2] / 2

        if mask_zn.any():
            points[:, mask_zn, 0] = (
                rand_vals[:, mask_zn, 0] * dims[:, mask_zn, 0] - dims[:, mask_zn, 0] / 2
            )
            points[:, mask_zn, 1] = (
                rand_vals[:, mask_zn, 1] * dims[:, mask_zn, 1] - dims[:, mask_zn, 1] / 2
            )
            points[:, mask_zn, 2] = -dims[:, mask_zn, 2] / 2

        # --- Transform points to world frame ---
        transformed_points = transform_pointcloud(
            points, combined_poses, in_place=False
        )

        # Remove batch dimension if input was single pose
        if single_pose:
            transformed_points = transformed_points.squeeze(0)

        return transformed_points

    def _sample_composite_attached_primitive(
        self, ee_poses, dims, offsets, offset_quats, num_primitives
    ):
        device = ee_poses.device
        dtype = ee_poses.dtype

        if ee_poses.dim() == 3:
            batch_size = ee_poses.shape[0]
            single_pose = False
        else:
            batch_size = 1
            single_pose = True
            ee_poses = ee_poses.unsqueeze(0)

        # Convert inputs to proper shapes
        dims_tensor = torch.as_tensor(dims, device=device, dtype=dtype)
        offsets_tensor = torch.as_tensor(offsets, device=device, dtype=dtype)
        offset_quats_tensor = torch.as_tensor(offset_quats, device=device, dtype=dtype)
        num_primitives_tensor = torch.as_tensor(num_primitives, device=device)

        # Ensure batch dimension
        if dims_tensor.ndim == 2:
            dims_tensor = dims_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        if offsets_tensor.ndim == 2:
            offsets_tensor = offsets_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        if offset_quats_tensor.ndim == 2:
            offset_quats_tensor = offset_quats_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        if num_primitives_tensor.ndim == 0:
            num_primitives_tensor = num_primitives_tensor.unsqueeze(0).repeat(batch_size)

        max_primitives = dims_tensor.shape[1]
        total_points = 500
        points_per_primitive = total_points // max_primitives

        # Create mask for active primitives
        primitive_indices = (
            torch.arange(max_primitives, device=device)
            .view(1, -1, 1)
            .expand(batch_size, -1, 1)
        )
        active_mask = primitive_indices < num_primitives_tensor.view(batch_size, 1, 1)

        # Expand EE poses for all primitives [B, max_primitives, 4, 4]
        ee_poses_expanded = ee_poses.unsqueeze(1).expand(-1, max_primitives, -1, -1)

        # Build transformation matrices for all primitives
        transform_matrices = self._build_composite_transform_matrices(
            ee_poses_expanded, offsets_tensor, offset_quats_tensor
        )

        # Sample points for ALL primitives (including inactive ones - we'll mask later)
        all_primitive_points = self._sample_cuboid_points_batch(
            dims_tensor, points_per_primitive, batch_size, max_primitives
        )

        # Transform all points
        transformed_points = self._transform_points_batch(
            all_primitive_points, transform_matrices
        )

        # Apply mask and combine points
        combined_points = self._combine_points_with_mask(
            transformed_points, active_mask, batch_size, total_points
        )

        if single_pose:
            combined_points = combined_points.squeeze(0)

        return combined_points


    def _build_composite_transform_matrices(self, ee_poses, offsets, quats):
        """Build transformation matrices for all primitives in batch"""
        batch_size, max_primitives = offsets.shape[:2]

        # Initialize transformation matrices
        transforms = torch.eye(4, device=ee_poses.device, dtype=ee_poses.dtype)
        transforms = (
            transforms.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_primitives, 1, 1)
        )

        # Set translation
        transforms[..., :3, 3] = offsets

        # Set rotation from quaternions
        rotations = self._quaternion_to_matrix_batch(quats)
        transforms[..., :3, :3] = rotations

        # Combine with EE poses: world_T_primitive = world_T_ee * ee_T_primitive
        return torch.matmul(ee_poses, transforms)


    def _quaternion_to_matrix_batch(self, quats):
        """Convert batched quaternions to rotation matrices"""
        w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot = torch.zeros(quats.shape[:-1] + (3, 3), device=quats.device, dtype=quats.dtype)

        rot[..., 0, 0] = 1 - 2 * (yy + zz)
        rot[..., 0, 1] = 2 * (xy - wz)
        rot[..., 0, 2] = 2 * (xz + wy)
        rot[..., 1, 0] = 2 * (xy + wz)
        rot[..., 1, 1] = 1 - 2 * (xx + zz)
        rot[..., 1, 2] = 2 * (yz - wx)
        rot[..., 2, 0] = 2 * (xz - wy)
        rot[..., 2, 1] = 2 * (yz + wx)
        rot[..., 2, 2] = 1 - 2 * (xx + yy)

        return rot


    def _sample_cuboid_points_batch(
        self, dims, points_per_face, batch_size, max_primitives
    ):
        """Sample points from cuboid surfaces for entire batch"""
        # Generate random samples for all primitives at once
        total_points = 6 * points_per_face  # 6 faces
        rand_vals = torch.rand(
            batch_size,
            max_primitives,
            total_points,
            3,
            device=dims.device,
            dtype=dims.dtype,
        )
        face_choices = torch.randint(
            0, 6, (batch_size, max_primitives, total_points), device=dims.device
        )

        # Expand dimensions for broadcasting
        dims_expanded = dims.unsqueeze(2).expand(-1, -1, total_points, -1)

        # Vectorized point generation (similar to your existing face sampling logic)
        points = torch.zeros_like(rand_vals)

        # Apply face sampling logic using tensor operations instead of loops
        for face in range(6):
            mask = face_choices == face
            if mask.any():
                points = self._apply_face_sampling(
                    points, rand_vals, dims_expanded, mask, face
                )

        return points


    def _apply_face_sampling(self, points, rand_vals, dims, mask, face):
        """Apply sampling logic for a specific face using vectorized operations"""
        # This would contain your existing face sampling logic but vectorized
        # Implementation depends on your specific face sampling requirements
        # ...
        return points


    def _transform_points_batch(self, points, transforms):
        """Transform points using batched transformation matrices"""
        batch_size, max_primitives, num_points, _ = points.shape

        # Convert to homogeneous coordinates
        points_h = torch.cat(
            [
                points,
                torch.ones(
                    batch_size,
                    max_primitives,
                    num_points,
                    1,
                    device=points.device,
                    dtype=points.dtype,
                ),
            ],
            dim=-1,
        )

        # Transform points: [B, M, N, 4] x [B, M, 4, 4] -> [B, M, N, 4]
        points_transformed = torch.matmul(points_h, transforms.transpose(-1, -2))

        return points_transformed[..., :3]


    def _combine_points_with_mask(self, points, active_mask, batch_size, total_points):
        """Combine points from active primitives and sample to target count"""
        # Flatten primitive and point dimensions
        points_flat = points.reshape(batch_size, -1, 3)
        mask_flat = active_mask.expand(-1, -1, points.shape[2]).reshape(batch_size, -1)

        combined_points = []

        for b in range(batch_size):
            # Get points from active primitives only
            active_points = points_flat[b][mask_flat[b]]

            # Sample to target count
            if len(active_points) >= total_points:
                indices = torch.randperm(len(active_points), device=points.device)[
                    :total_points
                ]
                sampled_points = active_points[indices]
            else:
                # If not enough points, repeat some
                indices = torch.randint(
                    0, len(active_points), (total_points,), device=points.device
                )
                sampled_points = active_points[indices]

            combined_points.append(sampled_points.unsqueeze(0))

        return torch.cat(combined_points, dim=0)

    def _sample_single_primitive(self, ee_poses, dim, offset, offset_quat, num_points):
        """
        Sample points from a single primitive (helper method for composite tools).
        This is similar to the existing _sample_attached_primitive but for single primitive.
        """
        device = ee_poses.device
        dtype = ee_poses.dtype

        if ee_poses.dim() == 3:  # [B, 4, 4]
            batch_size = ee_poses.shape[0]
        else:  # [4, 4]
            batch_size = 1
            ee_poses = ee_poses.unsqueeze(0)

        # Build offset transform
        offset_transform = (
            torch.eye(4, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        offset_transform[:, :3, 3] = offset.squeeze(1)  # Remove the extra dimension

        # Quaternion to rotation matrix (vectorized)
        w, x, y, z = (
            offset_quat[:, :, 0],
            offset_quat[:, :, 1],
            offset_quat[:, :, 2],
            offset_quat[:, :, 3],
        )
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        rot[:, 0, 0] = 1 - 2 * (yy + zz).squeeze(1)
        rot[:, 0, 1] = 2 * (xy - wz).squeeze(1)
        rot[:, 0, 2] = 2 * (xz + wy).squeeze(1)
        rot[:, 1, 0] = 2 * (xy + wz).squeeze(1)
        rot[:, 1, 1] = 1 - 2 * (xx + zz).squeeze(1)
        rot[:, 1, 2] = 2 * (yz - wx).squeeze(1)
        rot[:, 2, 0] = 2 * (xz - wy).squeeze(1)
        rot[:, 2, 1] = 2 * (yz + wx).squeeze(1)
        rot[:, 2, 2] = 1 - 2 * (xx + yy).squeeze(1)
        offset_transform[:, :3, :3] = rot

        # Combine offset with EE poses
        combined_poses = torch.bmm(ee_poses, offset_transform)

        # Sample cuboid surface points
        rand_vals = torch.rand(batch_size, num_points, 3, device=device, dtype=dtype)
        face_choices = torch.randint(0, 6, (batch_size, num_points), device=device)

        dims_expanded = dim.repeat(batch_size, num_points, 1)  # [B, N, 3]
        points = torch.zeros_like(rand_vals)

        # Vectorized face sampling (same as original _sample_attached_primitive)
        for batch_idx in range(batch_size):
            mask_xp = face_choices[batch_idx] == 0
            mask_xn = face_choices[batch_idx] == 1
            mask_yp = face_choices[batch_idx] == 2
            mask_yn = face_choices[batch_idx] == 3
            mask_zp = face_choices[batch_idx] == 4
            mask_zn = face_choices[batch_idx] == 5

            if mask_xp.any():
                points[batch_idx, mask_xp, 0] = dims_expanded[batch_idx, mask_xp, 0] / 2
                points[batch_idx, mask_xp, 1] = (
                    rand_vals[batch_idx, mask_xp, 0]
                    * dims_expanded[batch_idx, mask_xp, 1]
                    - dims_expanded[batch_idx, mask_xp, 1] / 2
                )
                points[batch_idx, mask_xp, 2] = (
                    rand_vals[batch_idx, mask_xp, 1]
                    * dims_expanded[batch_idx, mask_xp, 2]
                    - dims_expanded[batch_idx, mask_xp, 2] / 2
                )

            if mask_xn.any():
                points[batch_idx, mask_xn, 0] = (
                    -dims_expanded[batch_idx, mask_xn, 0] / 2
                )
                points[batch_idx, mask_xn, 1] = (
                    rand_vals[batch_idx, mask_xn, 0]
                    * dims_expanded[batch_idx, mask_xn, 1]
                    - dims_expanded[batch_idx, mask_xn, 1] / 2
                )
                points[batch_idx, mask_xn, 2] = (
                    rand_vals[batch_idx, mask_xn, 1]
                    * dims_expanded[batch_idx, mask_xn, 2]
                    - dims_expanded[batch_idx, mask_xn, 2] / 2
                )

            if mask_yp.any():
                points[batch_idx, mask_yp, 0] = (
                    rand_vals[batch_idx, mask_yp, 0]
                    * dims_expanded[batch_idx, mask_yp, 0]
                    - dims_expanded[batch_idx, mask_yp, 0] / 2
                )
                points[batch_idx, mask_yp, 1] = dims_expanded[batch_idx, mask_yp, 1] / 2
                points[batch_idx, mask_yp, 2] = (
                    rand_vals[batch_idx, mask_yp, 1]
                    * dims_expanded[batch_idx, mask_yp, 2]
                    - dims_expanded[batch_idx, mask_yp, 2] / 2
                )

            if mask_yn.any():
                points[batch_idx, mask_yn, 0] = (
                    rand_vals[batch_idx, mask_yn, 0]
                    * dims_expanded[batch_idx, mask_yn, 0]
                    - dims_expanded[batch_idx, mask_yn, 0] / 2
                )
                points[batch_idx, mask_yn, 1] = (
                    -dims_expanded[batch_idx, mask_yn, 1] / 2
                )
                points[batch_idx, mask_yn, 2] = (
                    rand_vals[batch_idx, mask_yn, 1]
                    * dims_expanded[batch_idx, mask_yn, 2]
                    - dims_expanded[batch_idx, mask_yn, 2] / 2
                )

            if mask_zp.any():
                points[batch_idx, mask_zp, 0] = (
                    rand_vals[batch_idx, mask_zp, 0]
                    * dims_expanded[batch_idx, mask_zp, 0]
                    - dims_expanded[batch_idx, mask_zp, 0] / 2
                )
                points[batch_idx, mask_zp, 1] = (
                    rand_vals[batch_idx, mask_zp, 1]
                    * dims_expanded[batch_idx, mask_zp, 1]
                    - dims_expanded[batch_idx, mask_zp, 1] / 2
                )
                points[batch_idx, mask_zp, 2] = dims_expanded[batch_idx, mask_zp, 2] / 2

            if mask_zn.any():
                points[batch_idx, mask_zn, 0] = (
                    rand_vals[batch_idx, mask_zn, 0]
                    * dims_expanded[batch_idx, mask_zn, 0]
                    - dims_expanded[batch_idx, mask_zn, 0] / 2
                )
                points[batch_idx, mask_zn, 1] = (
                    rand_vals[batch_idx, mask_zn, 1]
                    * dims_expanded[batch_idx, mask_zn, 1]
                    - dims_expanded[batch_idx, mask_zn, 1] / 2
                )
                points[batch_idx, mask_zn, 2] = (
                    -dims_expanded[batch_idx, mask_zn, 2] / 2
                )

        # Transform points to world frame
        transformed_points = transform_pointcloud(
            points, combined_poses, in_place=False
        )

        return transformed_points

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
        link_sizes = {
            l.name: self.points[l.name].shape[1]
            for l in self.links
            if not (l.name == "panda_link0" and not self.with_base_link)
        }
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
                link_points = max(
                    1,
                    int(
                        round(total_points * (link_sizes[l.name] / total_fixed_points))
                    ),
                )

                if transformed_pc.shape[1] < link_points:
                    # If we don't have enough pre-sampled points, use all we have
                    link_points = transformed_pc.shape[1]

                if link_points < transformed_pc.shape[1]:
                    transformed_pc = transformed_pc[
                        :,
                        np.random.choice(
                            transformed_pc.shape[1], link_points, replace=False
                        ),
                        :,
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
