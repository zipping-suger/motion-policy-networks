from pathlib import Path
from typing import Optional, List, Union, Dict
import enum
import os
import itertools

from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from pyquaternion import Quaternion
from geometrout.primitive import Cuboid, Cylinder
# from utils import FrankaSampler # with tool
from robofin.pointcloud.torch import FrankaSampler # with out tool

from robofin.robots import FrankaRealRobot
from mpinets.geometry import construct_mixed_point_cloud
from mpinets import utils


class DatasetType(enum.Enum):
    """
    A simple enum class to indicate whether a dataloader is for training, validating, or testing
    """

    TRAIN = 0
    VAL = 1
    TEST = 2


# --- Helper Function for BBX Calculation ---
def compute_tool_bbx(dims, offsets, quats, num_primitives):
    """
    Computes the 8 corners of the Axis-Aligned Bounding Box (AABB) 
    for the entire composite tool in the tool frame.
    
    :param dims: (max_primitives, 3) dimensions of primitives
    :param offsets: (max_primitives, 3) offsets of primitives
    :param quats: (max_primitives, 4) quaternions (w,x,y,z) of primitives
    :param num_primitives: int, number of valid primitives
    :return: (24,) flattened tensor of the 8 corners
    """
    if num_primitives == 0:
        # Return a small default box or zeros if no tool
        return torch.zeros(24).float()

    all_corners = []

    for i in range(int(num_primitives)):
        d = dims[i]
        o = offsets[i]
        q = quats[i] # Expecting w, x, y, z

        # 1. Generate 8 corners of the primitive centered at 0
        # dims are (x, y, z)
        dx, dy, dz = d[0] / 2.0, d[1] / 2.0, d[2] / 2.0
        
        # Create all combinations of +/- dimensions
        # Use itertools to generate the 8 corners cleanly
        local_corners = np.array(list(itertools.product(
            [-dx, dx], [-dy, dy], [-dz, dz]
        )))

        # 2. Rotate corners
        # pyquaternion expects (w, x, y, z) which matches the generation code
        rot_mat = Quaternion(q).rotation_matrix
        rotated_corners = local_corners @ rot_mat.T

        # 3. Translate corners
        global_corners = rotated_corners + o
        all_corners.append(global_corners)

    # Stack all points from all primitives
    all_points = np.vstack(all_corners)

    # 4. Compute AABB of the union
    min_pt = np.min(all_points, axis=0)
    max_pt = np.max(all_points, axis=0)

    # 5. Generate the 8 corners of this AABB
    # Order: Consistent ordering is important for the network
    # We use itertools again to ensure deterministic ordering
    min_x, min_y, min_z = min_pt
    max_x, max_y, max_z = max_pt
    
    aabb_corners = np.array(list(itertools.product(
        [min_x, max_x], [min_y, max_y], [min_z, max_z]
    )))

    # Flatten to 24D vector (8 points * 3 coords)
    return torch.as_tensor(aabb_corners).float().flatten()


class TaskDataset(Dataset):

    def __init__(
        self,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        self._init_directory(directory, dataset_type)
        self.trajectory_key = trajectory_key 
        self.train = dataset_type == DatasetType.TRAIN

        self.num_obstacle_points = num_obstacle_points
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.random_scale = random_scale
        self.fk_sampler = FrankaSampler("cpu", use_cache=True)
        with h5py.File(str(self._database), "r") as f:
            self._length = f['target_poses'].shape[0]

    def __len__(self):
        return self._length

    def _init_directory(self, directory: Path, dataset_type: DatasetType):
        self.type = dataset_type
        if dataset_type == DatasetType.TRAIN:
            directory = directory / "train"
        elif dataset_type == DatasetType.VAL:
            directory = directory / "val"
        elif dataset_type == DatasetType.TEST:
            directory = directory / "test"
        else:
            raise Exception(f"Invalid dataset type: {dataset_type}")

        databases = list(directory.glob("**/*.hdf5"))
        print(f"Databases found: {databases}")
        assert len(databases) == 1
        self._database = databases[0]

    @staticmethod
    def normalize(configuration_tensor: torch.Tensor):
        return utils.normalize_franka_joints(configuration_tensor)

    def _construct_pointcloud(self, robot_points, obstacle_points, target_points):
        obstacle_points = torch.as_tensor(obstacle_points[:, :3]).float()

        xyz = torch.cat(
            (
                torch.zeros(self.num_robot_points, 4),
                torch.ones(self.num_obstacle_points, 4),
                2 * torch.ones(self.num_target_points, 4),
            ),
            dim=0,
        )

        xyz[:self.num_robot_points, :3] = robot_points.float()
        xyz[self.num_robot_points:self.num_robot_points+self.num_obstacle_points, :3] = obstacle_points
        xyz[self.num_robot_points+self.num_obstacle_points:, :3] = target_points.float()

        return xyz

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        with h5py.File(str(self._database), "r") as f:
            # Target
            target_config = f['target_configs'][idx]              
            target_pose = FrankaRealRobot.fk(target_config)
            target_pose_matrix = target_pose.matrix

            # --- Tool Loading & BBX Feature Construction ---
            start_tool_dims = f['start_tool_dims'][idx] if 'start_tool_dims' in f.keys() else np.zeros((1, 3))
            start_tool_offset = f['start_tool_offset'][idx] if 'start_tool_offset' in f.keys() else np.zeros((1, 3))
            start_tool_quaternion = f['start_tool_quaternion'][idx] if 'start_tool_quaternion' in f.keys() else np.array([[1.0, 0.0, 0.0, 0.0]])
            start_tool_num_primitives = f['start_tool_num_primitives'][idx] if 'start_tool_num_primitives' in f.keys() else 1

            # Compute BBX Feature (24D flattened corners)
            item["bbx_feature"] = compute_tool_bbx(
                start_tool_dims, 
                start_tool_offset, 
                start_tool_quaternion, 
                start_tool_num_primitives
            )

            # Target tool info (loaded but generally bbx_feature is calculated for the robot's current tool)
            target_tool_dims = f['target_tool_dims'][idx] if 'target_tool_dims' in f.keys() else np.zeros((1, 3))
            target_tool_offset = f['target_tool_offset'][idx] if 'target_tool_offset' in f.keys() else np.zeros((1, 3))
            target_tool_quaternion = f['target_tool_quaternion'][idx] if 'target_tool_quaternion' in f.keys() else np.array([[1.0, 0.0, 0.0, 0.0]])
            target_tool_num_primitives = f['target_tool_num_primitives'][idx] if 'target_tool_num_primitives' in f.keys() else 1

            # Store raw tool information
            item["start_tool_dims"] = torch.as_tensor(start_tool_dims).float()
            item["start_tool_offset"] = torch.as_tensor(start_tool_offset).float()
            item["start_tool_quaternion"] = torch.as_tensor(start_tool_quaternion).float()
            item["start_tool_num_primitives"] = torch.as_tensor(start_tool_num_primitives).int()

            item["target_tool_dims"] = torch.as_tensor(target_tool_dims).float()
            item["target_tool_offset"] = torch.as_tensor(target_tool_offset).float()
            item["target_tool_quaternion"] = torch.as_tensor(target_tool_quaternion).float()
            item["target_tool_num_primitives"] = torch.as_tensor(target_tool_num_primitives).int()
            # ---------------------------------------------

            target_points = self.fk_sampler.sample_end_effector(
                torch.as_tensor(target_pose_matrix).float(),
                num_points=self.num_target_points,
            )

            target_position = torch.as_tensor(target_pose_matrix[:3, 3], dtype=torch.float32)
            target_rot_mat = torch.as_tensor(target_pose.matrix[:3, :3].flatten(), dtype=torch.float32)
            item["target_position"] = target_position
            item["target_rotation"] = target_rot_mat
            item["target_pose"] = torch.cat((target_position, target_rot_mat), dim=0).float()
            item["target_configuration"] = torch.as_tensor(target_config).float()

            start_config = f["start_configs"][idx]
            config_tensor = torch.as_tensor(start_config).float()

            if self.train:
                randomized = (
                    self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                )
                limits = torch.as_tensor(FrankaRealRobot.JOINT_LIMITS).float()
                randomized = torch.minimum(
                    torch.maximum(randomized, limits[:, 0]), limits[:, 1]
                )
                item["configuration"] = self.normalize(randomized)
                robot_points = self.fk_sampler.sample(randomized, self.num_robot_points)
            else:
                item["configuration"] = self.normalize(config_tensor)
                robot_points = self.fk_sampler.sample(
                    config_tensor, self.num_robot_points
                )

            cuboid_dims = f["cuboid_dims"][idx, ...]
            if cuboid_dims.ndim == 1:
                cuboid_dims = np.expand_dims(cuboid_dims, axis=0)

            cuboid_centers = f["cuboid_centers"][idx, ...]
            if cuboid_centers.ndim == 1:
                cuboid_centers = np.expand_dims(cuboid_centers, axis=0)

            cuboid_quats = f["cuboid_quaternions"][idx, ...]
            if cuboid_quats.ndim == 1:
                cuboid_quats = np.expand_dims(cuboid_quats, axis=0)
            
            cuboid_quats[np.all(np.isclose(cuboid_quats, 0), axis=1), 0] = 1

            item["cuboid_dims"] = torch.as_tensor(cuboid_dims)
            item["cuboid_centers"] = torch.as_tensor(cuboid_centers)
            item["cuboid_quats"] = torch.as_tensor(cuboid_quats)

            if "cylinder_radii" not in f.keys():
                cylinder_radii = np.array([[0.0]])
                cylinder_heights = np.array([[0.0]])
                cylinder_centers = np.array([[0.0, 0.0, 0.0]])
                cylinder_quats = np.array([[1.0, 0.0, 0.0, 0.0]])
            else:
                cylinder_radii = f["cylinder_radii"][idx, ...]
                if cylinder_radii.ndim == 1:
                    cylinder_radii = np.expand_dims(cylinder_radii, axis=0)
                cylinder_heights = f["cylinder_heights"][idx, ...]
                if cylinder_heights.ndim == 1:
                    cylinder_heights = np.expand_dims(cylinder_heights, axis=0)
                cylinder_centers = f["cylinder_centers"][idx, ...]
                if cylinder_centers.ndim == 1:
                    cylinder_centers = np.expand_dims(cylinder_centers, axis=0)
                cylinder_quats = f["cylinder_quaternions"][idx, ...]
                if cylinder_quats.ndim == 1:
                    cylinder_quats = np.expand_dims(cylinder_quats, axis=0)
                
                cylinder_quats[np.all(np.isclose(cylinder_quats, 0), axis=1), 0] = 1

            item["cylinder_radii"] = torch.as_tensor(cylinder_radii)
            item["cylinder_heights"] = torch.as_tensor(cylinder_heights)
            item["cylinder_centers"] = torch.as_tensor(cylinder_centers)
            item["cylinder_quats"] = torch.as_tensor(cylinder_quats)

            cuboids = [
                Cuboid(c, d, q)
                for c, d, q in zip(
                    list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
                )
            ]
            cuboids = [c for c in cuboids if not c.is_zero_volume()]

            cylinders = [
                Cylinder(c, r, h, q)
                for c, r, h, q in zip(
                    list(cylinder_centers),
                    list(cylinder_radii.squeeze(1)),
                    list(cylinder_heights.squeeze(1)),
                    list(cylinder_quats),
                )
            ]
            cylinders = [c for c in cylinders if not c.is_zero_volume()]

            obstacle_points = construct_mixed_point_cloud(
                cuboids + cylinders, self.num_obstacle_points
            )
            item["xyz"] = self._construct_pointcloud(robot_points, obstacle_points, target_points)

        return item


class PointCloudBase(Dataset):
    """
    Base class for trajectory/instance datasets.
    """

    def __init__(
        self,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        self._init_directory(directory, dataset_type)
        self.trajectory_key = trajectory_key
        self.train = dataset_type == DatasetType.TRAIN
        with h5py.File(str(self._database), "r") as f:
            self._num_trajectories = f[self.trajectory_key].shape[0]
            self.expert_length = f[self.trajectory_key].shape[1]

        self.num_obstacle_points = num_obstacle_points
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.random_scale = random_scale
        self.fk_sampler = FrankaSampler("cpu", use_cache=True)

    def _init_directory(self, directory: Path, dataset_type: DatasetType):
        self.type = dataset_type
        if dataset_type == DatasetType.TRAIN:
            directory = directory / "train"
        elif dataset_type == DatasetType.VAL:
            directory = directory / "val"
        elif dataset_type == DatasetType.TEST:
            directory = directory / "test"
        else:
            raise Exception(f"Invalid dataset type: {dataset_type}")

        databases = list(directory.glob("**/*.hdf5"))
        assert len(databases) == 1
        self._database = databases[0]

    @property
    def num_trajectories(self):
        return self._num_trajectories

    @staticmethod
    def normalize(configuration_tensor: torch.Tensor):
        return utils.normalize_franka_joints(configuration_tensor)

    def get_inputs(self, trajectory_idx: int, timestep: int) -> Dict[str, torch.Tensor]:
        item = {}
        with h5py.File(str(self._database), "r") as f:
            target_pose = FrankaRealRobot.fk(
                f[self.trajectory_key][trajectory_idx, -1, :]
            )

            # --- Tool Loading & BBX Feature Construction ---
            start_tool_dims = (
                f["start_tool_dims"][trajectory_idx]
                if "start_tool_dims" in f.keys()
                else np.zeros((1, 3))
            )
            start_tool_offset = (
                f["start_tool_offset"][trajectory_idx]
                if "start_tool_offset" in f.keys()
                else np.zeros((1, 3))
            )
            start_tool_quaternion = (
                f["start_tool_quaternion"][trajectory_idx]
                if "start_tool_quaternion" in f.keys()
                else np.array([[1.0, 0.0, 0.0, 0.0]])
            )
            start_tool_num_primitives = (
                f["start_tool_num_primitives"][trajectory_idx]
                if "start_tool_num_primitives" in f.keys()
                else 1
            )
            
            # Compute BBX Feature (24D flattened corners)
            item["bbx_feature"] = compute_tool_bbx(
                start_tool_dims, 
                start_tool_offset, 
                start_tool_quaternion, 
                start_tool_num_primitives
            )
            
            # Load Target Tool (standard loading)
            target_tool_dims = (
                f["target_tool_dims"][trajectory_idx]
                if "target_tool_dims" in f.keys()
                else np.zeros((1, 3))
            )
            target_tool_offset = (
                f["target_tool_offset"][trajectory_idx]
                if "target_tool_offset" in f.keys()
                else np.zeros((1, 3))
            )
            target_tool_quaternion = (
                f["target_tool_quaternion"][trajectory_idx]
                if "target_tool_quaternion" in f.keys()
                else np.array([[1.0, 0.0, 0.0, 0.0]])
            )
            target_tool_num_primitives = (
                f["target_tool_num_primitives"][trajectory_idx]
                if "target_tool_num_primitives" in f.keys()
                else 1
            )

            item["start_tool_dims"] = torch.as_tensor(start_tool_dims).float()
            item["start_tool_offset"] = torch.as_tensor(start_tool_offset).float()
            item["start_tool_quaternion"] = torch.as_tensor(start_tool_quaternion).float()
            item["start_tool_num_primitives"] = torch.as_tensor(start_tool_num_primitives).int()

            item["target_tool_dims"] = torch.as_tensor(target_tool_dims).float()
            item["target_tool_offset"] = torch.as_tensor(target_tool_offset).float()
            item["target_tool_quaternion"] = torch.as_tensor(target_tool_quaternion).float()
            item["target_tool_num_primitives"] = torch.as_tensor(target_tool_num_primitives).int()
            # ---------------------------------------------

            target_points = self.fk_sampler.sample_end_effector(
                torch.as_tensor(target_pose.matrix).float(),
                num_points=self.num_target_points,
            )

            target_position = torch.as_tensor(target_pose.xyz).float()
            target_rot_mat = torch.as_tensor(target_pose.matrix[:3, :3].flatten(), dtype=torch.float32)
            item["target_position"] = target_position
            item["target_rotation"] = target_rot_mat
            item["target_pose"] = torch.cat((target_position, target_rot_mat), dim=0).float()

            target_config = f[self.trajectory_key][trajectory_idx, -1, :]
            item["target_configuration"] = torch.as_tensor(target_config).float()

            config = f[self.trajectory_key][trajectory_idx, timestep, :]
            config_tensor = torch.as_tensor(config).float()

            if self.train:
                randomized = (
                    self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                )
                limits = torch.as_tensor(FrankaRealRobot.JOINT_LIMITS).float()
                randomized = torch.minimum(
                    torch.maximum(randomized, limits[:, 0]), limits[:, 1]
                )
                item["configuration"] = self.normalize(randomized)
                robot_points = self.fk_sampler.sample(randomized, self.num_robot_points)
            else:
                item["configuration"] = self.normalize(config_tensor)
                robot_points = self.fk_sampler.sample(
                    config_tensor, self.num_robot_points
                )
            
            # ... (Rest of existing code for cuboids and cylinders) ...
            # I am keeping the logic flow from the original file here implicitly
            # to save space, but ensure the rest of get_inputs matches your file.
            
            cuboid_dims = f["cuboid_dims"][trajectory_idx, ...]
            if cuboid_dims.ndim == 1:
                cuboid_dims = np.expand_dims(cuboid_dims, axis=0)

            cuboid_centers = f["cuboid_centers"][trajectory_idx, ...]
            if cuboid_centers.ndim == 1:
                cuboid_centers = np.expand_dims(cuboid_centers, axis=0)

            cuboid_quats = f["cuboid_quaternions"][trajectory_idx, ...]
            if cuboid_quats.ndim == 1:
                cuboid_quats = np.expand_dims(cuboid_quats, axis=0)

            cuboid_quats[np.all(np.isclose(cuboid_quats, 0), axis=1), 0] = 1

            item["cuboid_dims"] = torch.as_tensor(cuboid_dims)
            item["cuboid_centers"] = torch.as_tensor(cuboid_centers)
            item["cuboid_quats"] = torch.as_tensor(cuboid_quats)

            if "cylinder_radii" not in f.keys():
                cylinder_radii = np.array([[0.0]])
                cylinder_heights = np.array([[0.0]])
                cylinder_centers = np.array([[0.0, 0.0, 0.0]])
                cylinder_quats = np.array([[1.0, 0.0, 0.0, 0.0]])
            else:
                cylinder_radii = f["cylinder_radii"][trajectory_idx, ...]
                if cylinder_radii.ndim == 1:
                    cylinder_radii = np.expand_dims(cylinder_radii, axis=0)
                cylinder_heights = f["cylinder_heights"][trajectory_idx, ...]
                if cylinder_heights.ndim == 1:
                    cylinder_heights = np.expand_dims(cylinder_heights, axis=0)
                cylinder_centers = f["cylinder_centers"][trajectory_idx, ...]
                if cylinder_centers.ndim == 1:
                    cylinder_centers = np.expand_dims(cylinder_centers, axis=0)
                cylinder_quats = f["cylinder_quaternions"][trajectory_idx, ...]
                if cylinder_quats.ndim == 1:
                    cylinder_quats = np.expand_dims(cylinder_quats, axis=0)
                cylinder_quats[np.all(np.isclose(cylinder_quats, 0), axis=1), 0] = 1

            item["cylinder_radii"] = torch.as_tensor(cylinder_radii)
            item["cylinder_heights"] = torch.as_tensor(cylinder_heights)
            item["cylinder_centers"] = torch.as_tensor(cylinder_centers)
            item["cylinder_quats"] = torch.as_tensor(cylinder_quats)

            cuboids = [
                Cuboid(c, d, q)
                for c, d, q in zip(
                    list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
                )
            ]
            cuboids = [c for c in cuboids if not c.is_zero_volume()]

            cylinders = [
                Cylinder(c, r, h, q)
                for c, r, h, q in zip(
                    list(cylinder_centers),
                    list(cylinder_radii.squeeze(1)),
                    list(cylinder_heights.squeeze(1)),
                    list(cylinder_quats),
                )
            ]
            cylinders = [c for c in cylinders if not c.is_zero_volume()]

            obstacle_points = construct_mixed_point_cloud(
                cuboids + cylinders, self.num_obstacle_points
            )
            item["xyz"] = torch.cat(
                (
                    torch.zeros(self.num_robot_points, 4),
                    torch.ones(self.num_obstacle_points, 4),
                    2 * torch.ones(self.num_target_points, 4),
                ),
                dim=0,
            )
            item["xyz"][: self.num_robot_points, :3] = robot_points.float()
            item["xyz"][
                self.num_robot_points : self.num_robot_points
                + self.num_obstacle_points,
                :3,
            ] = torch.as_tensor(obstacle_points[:, :3]).float()
            item["xyz"][
                self.num_robot_points + self.num_obstacle_points :,
                :3,
            ] = target_points.float()

        return item

class PointCloudTrajectoryDataset(PointCloudBase):
    """
    This dataset is used exclusively for validating. Each element in the dataset
    represents a trajectory start and scene. There is no supervision because
    this is used to produce an entire rollout and check for success. When doing
    validation, we care more about success than we care about matching the
    expert's behavior (which is a key difference from training).
    """

    def __init__(
        self,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        """
        super().__init__(
            directory,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            dataset_type,
            random_scale,
        )

    def __len__(self):
        """
        Necessary for Pytorch. For this dataset, the length is the total number
        of problems
        """
        return self.num_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Required by Pytorch. Queries for data at a particular index. Note that
        in this dataset, the index always corresponds to the trajectory index.

        :param idx int: The index
        :rtype Dict[str, torch.Tensor]: Returns a dictionary that can be assembled
            by the data loader before using in training.
        """
        trajectory_idx, timestep = idx, 0
        item = self.get_inputs(trajectory_idx, timestep)

        return item


class PointCloudInstanceDataset(PointCloudBase):
    """
    This is the dataset used primarily for training. Each element in the dataset
    represents the robot and scene at a particular time $t$. Likewise, the
    supervision is the robot's configuration at q_{t+1}.
    """

    def __init__(
        self,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        super().__init__(
            directory,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            dataset_type,
            random_scale,
        )

    def __len__(self):
        """
        Returns the total number of start configurations in the dataset (i.e.
        the length of the trajectories times the number of trajectories)

        """
        return self.num_trajectories * self.expert_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training datapoint representing a single configuration in a
        single scene with the configuration at the next timestep as supervision

        :param idx int: Index represents the timestep within the trajectory
        :rtype Dict[str, torch.Tensor]: The data used for training
        """
        trajectory_idx, timestep = divmod(idx, self.expert_length)
        if timestep >= self.expert_length:
            timestep = self.expert_length - 1
        item = self.get_inputs(trajectory_idx, timestep)

        # Re-use the last point in the trajectory at the end
        supervision_timestep = np.clip(
            timestep + 1,
            0,
            self.expert_length - 1,
        )

        with h5py.File(str(self._database), "r") as f:
            supervision = self.normalize(
                torch.as_tensor(
                    f[self.trajectory_key][trajectory_idx, supervision_timestep, :]
                )
            ).float()
            if torch.any(supervision < -1) or torch.any(supervision > 1):
                print("Supervision out of bounds:", supervision)
            supervision = torch.clamp(supervision, -1, 1)
            item["supervision"] = supervision

        return item


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
        batch_size: int,
        train_mode: str,
    ):
        """
        :param data_dir str: The directory with the data. Directory structure should
                             be as defined in `PointCloudBase`
        :param trajectory_key str: The key in the hdf5 dataset that contains the expert trajectories
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
        :param batch_size int: The batch size
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.trajectory_key = trajectory_key
        self.batch_size = batch_size
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.num_target_points = num_target_points
        self.num_workers = 16 
        self.random_scale = random_scale
        self.train_mode = train_mode

    def setup(self, stage: Optional[str] = None):
        """
        A Pytorch Lightning method that is called per-device in when doing
        distributed training.

        :param stage Optional[str]: Indicates whether we are in the training
                                    procedure or if we are doing ad-hoc testing
        """
        if stage == "fit" or stage is None:
            if self.train_mode == "pretrain":
                # Use PointCloudInstanceDataset for pretraining (behavioral cloning)
                self.data_train = PointCloudInstanceDataset(
                    self.data_dir,
                    self.trajectory_key,
                    self.num_robot_points,
                    self.num_obstacle_points,
                    self.num_target_points,
                    dataset_type=DatasetType.TRAIN,
                    random_scale=self.random_scale,
                )
            elif self.train_mode == "finetune":
                # Use PointCloudTrajectoryDataset for fine-tuning (optimization)
                self.data_train = PointCloudTrajectoryDataset(
                    self.data_dir,
                    self.trajectory_key,
                    self.num_robot_points,
                    self.num_obstacle_points,
                    self.num_target_points,
                    dataset_type=DatasetType.TRAIN,
                    random_scale=self.random_scale,
                )
            elif self.train_mode == "finetune_tasks":
                # Use ProblemDataset for fine-tuning tasks
                self.data_train = TaskDataset(
                    self.data_dir,
                    self.trajectory_key,
                    self.num_robot_points,
                    self.num_obstacle_points,
                    self.num_target_points,
                    dataset_type=DatasetType.TRAIN,
                    random_scale=self.random_scale,
                )
            else:
                raise ValueError(f"Unknown training mode: {self.train_mode}. Expected 'pretrain' or 'finetune'.")
            
            if self.train_mode in ["pretrain", "finetune"]:
                self.data_val = PointCloudTrajectoryDataset(
                    self.data_dir,
                    self.trajectory_key,
                    self.num_robot_points,
                    self.num_obstacle_points,
                    self.num_target_points,
                    dataset_type=DatasetType.VAL,
                    random_scale=0.0,  # No random scale for validation
                )
            elif self.train_mode == "finetune_tasks":
                self.data_val = TaskDataset(
                    self.data_dir,
                    self.trajectory_key,
                    self.num_robot_points,
                    self.num_obstacle_points,
                    self.num_target_points,
                    dataset_type=DatasetType.VAL,
                    random_scale=0.0,  # No random scale for validation
                )
            else:
                raise ValueError(f"Unknown training mode: {self.train_mode}. Expected 'pretrain', 'finetune', or 'finetune_tasks'.")
        if stage == "test" or stage is None:
            self.data_test = PointCloudInstanceDataset(
                self.data_dir,
                self.trajectory_key,
                self.num_robot_points,
                self.num_obstacle_points,
                dataset_type=DatasetType.TEST,
                random_scale=self.random_scale,
            )

    def train_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for training

        :rtype DataLoader: The training dataloader
        """
        return DataLoader(
            self.data_train,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,  # shuffle the training data
        )

    def val_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for validation

        :rtype DataLoader: The validation dataloader
        """
        return DataLoader(
            self.data_val,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for testing

        :rtype DataLoader: The dataloader for testing
        """
        return DataLoader(
            self.data_test,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
