from typing import Union, Tuple
import numpy as np
import torch
from robofin.robots import FrankaRobot, FrankaRealRobot
import trimesh
from robofin.torch_urdf import TorchURDF
import logging
from pathlib import Path


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
    Optimized FrankaSampler with precomputation and caching for primitive point clouds.
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
        self.primitive_cache = {}  # Cache for primitive point clouds
        self.transform_cache = {}  # Cache for transformation matrices
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

    def _get_primitive_cache_key(self, dims, num_points):
        """Generate cache key for primitive dimensions and point count"""
        # Round dimensions to avoid floating point precision issues
        dim_key = tuple(dims.cpu().numpy().flatten().round(4))
        return (dim_key, num_points)

    def _sample_cuboid_points_fast(self, dims, num_points):
        """Fast cuboid surface sampling with caching"""
        cache_key = self._get_primitive_cache_key(dims, num_points)

        if cache_key in self.primitive_cache:
            return self.primitive_cache[cache_key].clone()

        device = dims.device
        dtype = dims.dtype

        # Vectorized sampling
        rand_vals = torch.rand(num_points, 3, device=device, dtype=dtype)
        face_choices = torch.randint(0, 6, (num_points,), device=device)

        points = torch.zeros(num_points, 3, device=device, dtype=dtype)
        dims_expanded = dims.unsqueeze(0).expand(num_points, -1)

        # Vectorized face assignments
        mask_xp = face_choices == 0
        mask_xn = face_choices == 1
        mask_yp = face_choices == 2
        mask_yn = face_choices == 3
        mask_zp = face_choices == 4
        mask_zn = face_choices == 5

        if mask_xp.any():
            points[mask_xp, 0] = dims_expanded[mask_xp, 0] / 2
            points[mask_xp, 1] = (
                rand_vals[mask_xp, 0] * dims_expanded[mask_xp, 1]
                - dims_expanded[mask_xp, 1] / 2
            )
            points[mask_xp, 2] = (
                rand_vals[mask_xp, 1] * dims_expanded[mask_xp, 2]
                - dims_expanded[mask_xp, 2] / 2
            )

        if mask_xn.any():
            points[mask_xn, 0] = -dims_expanded[mask_xn, 0] / 2
            points[mask_xn, 1] = (
                rand_vals[mask_xn, 0] * dims_expanded[mask_xn, 1]
                - dims_expanded[mask_xn, 1] / 2
            )
            points[mask_xn, 2] = (
                rand_vals[mask_xn, 1] * dims_expanded[mask_xn, 2]
                - dims_expanded[mask_xn, 2] / 2
            )

        if mask_yp.any():
            points[mask_yp, 0] = (
                rand_vals[mask_yp, 0] * dims_expanded[mask_yp, 0]
                - dims_expanded[mask_yp, 0] / 2
            )
            points[mask_yp, 1] = dims_expanded[mask_yp, 1] / 2
            points[mask_yp, 2] = (
                rand_vals[mask_yp, 1] * dims_expanded[mask_yp, 2]
                - dims_expanded[mask_yp, 2] / 2
            )

        if mask_yn.any():
            points[mask_yn, 0] = (
                rand_vals[mask_yn, 0] * dims_expanded[mask_yn, 0]
                - dims_expanded[mask_yn, 0] / 2
            )
            points[mask_yn, 1] = -dims_expanded[mask_yn, 1] / 2
            points[mask_yn, 2] = (
                rand_vals[mask_yn, 1] * dims_expanded[mask_yn, 2]
                - dims_expanded[mask_yn, 2] / 2
            )

        if mask_zp.any():
            points[mask_zp, 0] = (
                rand_vals[mask_zp, 0] * dims_expanded[mask_zp, 0]
                - dims_expanded[mask_zp, 0] / 2
            )
            points[mask_zp, 1] = (
                rand_vals[mask_zp, 1] * dims_expanded[mask_zp, 1]
                - dims_expanded[mask_zp, 1] / 2
            )
            points[mask_zp, 2] = dims_expanded[mask_zp, 2] / 2

        if mask_zn.any():
            points[mask_zn, 0] = (
                rand_vals[mask_zn, 0] * dims_expanded[mask_zn, 0]
                - dims_expanded[mask_zn, 0] / 2
            )
            points[mask_zn, 1] = (
                rand_vals[mask_zn, 1] * dims_expanded[mask_zn, 1]
                - dims_expanded[mask_zn, 1] / 2
            )
            points[mask_zn, 2] = -dims_expanded[mask_zn, 2] / 2

        self.primitive_cache[cache_key] = points.clone()
        return points

    def _build_batch_transforms(self, offsets, quats):
        """Build transformation matrices in batch"""
        device = offsets.device
        dtype = offsets.dtype

        batch_size = offsets.shape[0]

        # Create base transformation matrices
        transforms = (
            torch.eye(4, device=device, dtype=dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # Set translations
        transforms[:, :3, 3] = offsets

        # Convert quaternions to rotation matrices (batched)
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

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

        transforms[:, :3, :3] = rot
        return transforms

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
        Optimized composite tool sampling with batched operations and caching
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

        # Add attached composite tool points if specified
        if tool_num_primitives.ndim == 0:
            if tool_num_primitives > 0:
                ee_pose = self.end_effector_pose(config, frame="right_gripper")
                primitive_points = self._sample_composite_attached_primitive_fast(
                    ee_pose, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)
        else:
            if torch.any(tool_num_primitives > 0):
                ee_pose = self.end_effector_pose(config, frame="right_gripper")
                primitive_points = self._sample_composite_attached_primitive_fast(
                    ee_pose, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)

        if num_points is None:
            return pc
        return pc[:, torch.randperm(pc.shape[1])[:num_points], :]

    def _sample_composite_attached_primitive_fast(
        self, ee_poses, dims, offsets, quats, num_primitives
    ):
        """
        Fast batched composite tool sampling with caching
        """
        device = ee_poses.device
        dtype = ee_poses.dtype

        if ee_poses.dim() == 3:
            batch_size = ee_poses.shape[0]
            single_pose = False
        else:
            batch_size = 1
            single_pose = True
            ee_poses = ee_poses.unsqueeze(0)

        # Convert inputs to batched tensors
        dims_tensor = torch.as_tensor(dims, device=device, dtype=dtype)
        offsets_tensor = torch.as_tensor(offsets, device=device, dtype=dtype)
        quats_tensor = torch.as_tensor(quats, device=device, dtype=dtype)
        num_primitives_tensor = torch.as_tensor(num_primitives, device=device)

        # Ensure batch dimension
        if dims_tensor.ndim == 2:
            dims_tensor = dims_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        if offsets_tensor.ndim == 2:
            offsets_tensor = offsets_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        if quats_tensor.ndim == 2:
            quats_tensor = quats_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
        if num_primitives_tensor.ndim == 0:
            num_primitives_tensor = num_primitives_tensor.unsqueeze(0).repeat(
                batch_size
            )

        max_primitives = dims_tensor.shape[1]
        total_points = 500
        points_per_primitive = total_points // max_primitives
        remainder_points = total_points - (points_per_primitive * (max_primitives - 1))

        all_primitive_points = []

        # Process each primitive in batch
        for prim_idx in range(max_primitives):
            # Get parameters for this primitive across all batches
            prim_dims = dims_tensor[:, prim_idx]
            prim_offsets = offsets_tensor[:, prim_idx]
            prim_quats = quats_tensor[:, prim_idx]

            # Skip zero primitives
            zero_mask = torch.all(prim_dims == 0, dim=1)
            if torch.all(zero_mask):
                # Add placeholder zeros
                placeholder = torch.zeros(
                    batch_size, points_per_primitive, 3, device=device, dtype=dtype
                )
                all_primitive_points.append(placeholder)
                continue

            # Build transforms for this primitive
            transform_mats = self._build_batch_transforms(prim_offsets, prim_quats)
            combined_poses = torch.bmm(ee_poses, transform_mats)

            # Determine points for this primitive
            current_points = (
                remainder_points
                if prim_idx == max_primitives - 1
                else points_per_primitive
            )

            # Sample points using cache (use first batch element's dimensions as key)
            cache_dims = prim_dims[0] if batch_size > 0 else prim_dims
            primitive_template = self._sample_cuboid_points_fast(
                cache_dims, current_points
            )

            # Expand to batch size
            primitive_points = primitive_template.unsqueeze(0).repeat(batch_size, 1, 1)

            # Transform points
            transformed_points = transform_pointcloud(
                primitive_points, combined_poses, in_place=False
            )

            all_primitive_points.append(transformed_points)

        # Combine all primitive points
        if all_primitive_points:
            result = torch.cat(all_primitive_points, dim=1)

            # Ensure exactly total_points
            if result.shape[1] > total_points:
                indices = torch.stack(
                    [
                        torch.randperm(result.shape[1])[:total_points]
                        for _ in range(batch_size)
                    ]
                )
                result = torch.gather(
                    result, 1, indices.unsqueeze(-1).expand(-1, -1, 3)
                )
            elif result.shape[1] < total_points:
                padding = torch.zeros(
                    batch_size,
                    total_points - result.shape[1],
                    3,
                    device=device,
                    dtype=dtype,
                )
                result = torch.cat([result, padding], dim=1)
        else:
            result = torch.zeros(
                batch_size, total_points, 3, device=device, dtype=dtype
            )

        if single_pose:
            result = result.squeeze(0)

        return result

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
        Optimized composite end effector sampling
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
        if tool_num_primitives.ndim == 0:
            if tool_num_primitives > 0:
                primitive_points = self._sample_composite_attached_primitive_fast(
                    poses, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)
        else:
            if torch.any(tool_num_primitives > 0):
                primitive_points = self._sample_composite_attached_primitive_fast(
                    poses, tool_dims, tool_offsets, tool_quats, tool_num_primitives
                )
                pc = torch.cat([pc, primitive_points], dim=1)

        if num_points is None:
            return pc
        return pc[:, torch.randperm(pc.shape[1])[:num_points], :]

    def sample_per_link(self, config, total_points=None):
        """
        Samples points from each link's surface separately
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

            link_pc = self.points[l.name].float().repeat((values[idx].shape[0], 1, 1))
            transformed_pc = transform_pointcloud(
                link_pc,
                values[idx],
                in_place=True,
            )

            if total_points is not None:
                link_points = max(
                    1,
                    int(
                        round(total_points * (link_sizes[l.name] / total_fixed_points))
                    ),
                )

                if transformed_pc.shape[1] < link_points:
                    link_points = transformed_pc.shape[1]

                if link_points < transformed_pc.shape[1]:
                    transformed_pc = transformed_pc[
                        :,
                        torch.randperm(transformed_pc.shape[1])[:link_points],
                        :,
                    ]

            if squeeze_output:
                transformed_pc = transformed_pc.squeeze(0)

            per_link_pcs[l.name] = transformed_pc

        return per_link_pcs

    def clear_cache(self):
        """Clear the primitive cache to free memory"""
        self.primitive_cache.clear()
        self.transform_cache.clear()


# Keep the existing normalization functions unchanged
def _normalize_franka_joints_numpy(
    batch_trajectory: np.ndarray,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> np.ndarray:
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
