import torch
from torch import nn
from robofin.pointcloud.torch import FrankaCollisionSampler
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from mpinets.utils import FrankaSampler

from mpinets import loss
from mpinets.utils import unnormalize_franka_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Tuple, Sequence, Dict, Callable


class MotionPolicyNetwork(pl.LightningModule):
    """
    The architecture laid out here is the default architecture laid out in the
    Motion Policy Networks paper (Fishman, et. al, 2022).
    """

    def __init__(self):
        """
        Constructs the model
        """
        super().__init__()
        self.point_cloud_encoder = MPiNetsPointNet()
        self.config_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(12, 32),  # 12 for pos + rotation matrix
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        
        # --- NEW: Bounding Box Encoder ---
        # Input: 24 (8 corners * 3 coordinates)
        # Output: 64 (to match config and target encoders)
        self.bbx_encoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )

        # --- UPDATED: Decoder Input Dimension ---
        # Previous: 1024 (PC) + 64 (Config) + 64 (Target) = 1152
        # New: 1024 (PC) + 64 (Config) + 64 (Target) + 64 (BBX) = 1216
        self.decoder = nn.Sequential(
            nn.Linear(1024 + 64 + 64 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 7),
        )

    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, xyz: torch.Tensor, q: torch.Tensor, target: torch.Tensor, bbx_feature: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Passes data through the network to produce an output

        :param xyz torch.Tensor: Tensor representing the point cloud [B x N x 4]
        :param q torch.Tensor: The current robot configuration [B x 7]
        :param target torch.Tensor: The target pose [B x 12]
        :param bbx_feature torch.Tensor: The tool bounding box corners [B x 24]
        :rtype torch.Tensor: The displacement [B x 7]
        """
        pc_encoding = self.point_cloud_encoder(xyz)
        config_encoding = self.config_encoder(q)
        target_encoding = self.target_encoder(target)
        bbx_encoding = self.bbx_encoder(bbx_feature)
        
        # Concatenate all features
        x = torch.cat((pc_encoding, config_encoding, target_encoding, bbx_encoding), dim=1)
        return self.decoder(x)


class TrainingMotionPolicyNetwork(MotionPolicyNetwork):
    """
    An version of the MotionPolicyNetwork model that has additional attributes
    necessary during training.
    """

    def __init__(
        self,
        num_robot_points: int,
        point_match_loss_weight: float,
        collision_loss_weight: float,
    ):
        super().__init__()
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.fk_sampler = None
        self.collision_sampler = None
        self.loss_fun = loss.CollisionAndBCLossContainer()

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        unnormalize: bool = False,
    ) -> List[torch.Tensor]:
        """
        Rolls out the policy an arbitrary length by calling it iteratively
        """
        xyz, q, target_pose = (
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
        )
        
        # --- NEW: Extract BBX Feature ---
        # Default to zeros if not present for compatibility
        bbx_feature = batch.get("bbx_feature", torch.zeros((xyz.shape[0], 24), device=xyz.device))

        # Get composite tool parameters
        start_tool_dims = batch.get(
            "start_tool_dims", torch.zeros((1, 1, 3), device=xyz.device)
        )
        start_tool_offset = batch.get(
            "start_tool_offset", torch.zeros((1, 1, 3), device=xyz.device)
        )
        start_tool_quaternion = batch.get(
            "start_tool_quaternion",
            torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=xyz.device),
        )
        start_tool_num_primitives = batch.get(
            "start_tool_num_primitives", torch.ones(xyz.shape[0], device=xyz.device)
        )

        # This block is to adapt for the case where we only want to roll out a
        # single trajectory
        if q.ndim == 1:
            xyz = xyz.unsqueeze(0)
            q = q.unsqueeze(0)
            bbx_feature = bbx_feature.unsqueeze(0) # Unsqueeze BBX as well
            start_tool_dims = start_tool_dims.unsqueeze(0)
            start_tool_offset = start_tool_offset.unsqueeze(0)
            start_tool_quaternion = start_tool_quaternion.unsqueeze(0)
            start_tool_num_primitives = start_tool_num_primitives.unsqueeze(0)

        if unnormalize:
            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            trajectory = [q_unnorm]
        else:
            trajectory = [q]

        for i in range(rollout_length):
            # --- UPDATED: Pass bbx_feature to forward() ---
            q = torch.clamp(q + self(xyz, q, target_pose, bbx_feature), min=-1, max=1)
            
            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            q_unnorm = q_unnorm.type_as(q)
            if unnormalize:
                trajectory.append(q_unnorm)
            else:
                trajectory.append(q)

            # Use composite sampling for robot point cloud
            samples = sampler(
                q_unnorm,
                start_tool_dims,
                start_tool_offset,
                start_tool_quaternion,
                start_tool_num_primitives,
            ).type_as(xyz)

            xyz[:, : samples.shape[1], :3] = samples

        return trajectory

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        A function called automatically by Pytorch Lightning during training.
        """
        xyz, q, target_pose = (
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
        )
        
        # --- NEW: Extract BBX Feature ---
        bbx_feature = batch.get("bbx_feature", torch.zeros((xyz.shape[0], 24), device=xyz.device))

        # --- UPDATED: Pass bbx_feature to forward() ---
        y_hat = torch.clamp(q + self(xyz, q, target_pose, bbx_feature), min=-1, max=1)

        (
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
            supervision,
        ) = (
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
            batch["supervision"],
        )

        (
            start_tool_dims,
            start_tool_offset,
            start_tool_quaternion,
            start_tool_num_primitives,
        ) = (
            batch.get(
                "start_tool_dims", torch.zeros((xyz.shape[0], 1, 3), device=xyz.device)
            ),
            batch.get(
                "start_tool_offset",
                torch.zeros((xyz.shape[0], 1, 3), device=xyz.device),
            ),
            batch.get(
                "start_tool_quaternion",
                torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=xyz.device).repeat(
                    xyz.shape[0], 1, 1
                ),
            ),
            batch.get(
                "start_tool_num_primitives", torch.ones(xyz.shape[0], device=xyz.device)
            ),
        )

        collision_loss, point_match_loss = self.loss_fun(
            y_hat,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
            start_tool_dims,
            start_tool_offset,
            start_tool_quaternion,
            start_tool_num_primitives,
            supervision,
        )

        self.log("point_match_loss", point_match_loss)
        self.log("collision_loss", collision_loss)
        train_loss = (
            self.point_match_loss_weight * point_match_loss
            + self.collision_loss_weight * collision_loss
        )
        self.log("train_loss", train_loss)
        return train_loss

    def sample(
        self,
        q: torch.Tensor,
        start_tool_dims: torch.Tensor,
        start_tool_offset: torch.Tensor,
        start_tool_quaternion: torch.Tensor,
        start_tool_num_primitives: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links
        """
        assert self.fk_sampler is not None

        # Handle batch dimension for tool parameters
        if (
            start_tool_dims.ndim == 3
            and start_tool_dims.shape[0] == 1
            and q.shape[0] > 1
        ):
            start_tool_dims = start_tool_dims.repeat(q.shape[0], 1, 1)
        if (
            start_tool_offset.ndim == 3
            and start_tool_offset.shape[0] == 1
            and q.shape[0] > 1
        ):
            start_tool_offset = start_tool_offset.repeat(q.shape[0], 1, 1)
        if (
            start_tool_quaternion.ndim == 3
            and start_tool_quaternion.shape[0] == 1
            and q.shape[0] > 1
        ):
            start_tool_quaternion = start_tool_quaternion.repeat(q.shape[0], 1, 1)

        return self.fk_sampler.sample_composite(
            q,
            start_tool_dims,
            start_tool_offset,
            start_tool_quaternion,
            start_tool_num_primitives,
            self.num_robot_points,
        )

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        This is a Pytorch Lightning function run automatically across devices
        during the validation loop
        """

        # These are defined here because they need to be set on the correct devices.
        # The easiest way to do this is to do it at call-time
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        if self.collision_sampler is None:
            self.collision_sampler = FrankaCollisionSampler(
                self.device, with_base_link=False
            )
        rollout = self.rollout(batch, 69, self.sample, unnormalize=True)

        assert self.fk_sampler is not None  # Necessary for mypy to type properly

        # Use standard end effector calculation (tool doesn't affect end effector frame)
        eff = self.fk_sampler.end_effector_pose(rollout[-1])
        position_error = torch.linalg.vector_norm(
            eff[:, :3, -1] - batch["target_position"], dim=1
        )
        avg_target_error = torch.mean(position_error)

        cuboids = TorchCuboids(
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
        )
        cylinders = TorchCylinders(
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )

        B = batch["cuboid_centers"].size(0)
        rollout = torch.stack(rollout, dim=1)
        # Here is some Pytorch broadcasting voodoo to calculate whether each
        # rollout has a collision or not (looking to calculate the collision rate)
        assert rollout.shape == (B, 70, 7)
        rollout = rollout.reshape(-1, 7)
        has_collision = torch.zeros(B, dtype=torch.bool, device=self.device)
        collision_spheres = self.collision_sampler.compute_spheres(rollout)
        for radius, spheres in collision_spheres:
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((B, -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert sdf_values.shape == (B, 70, num_spheres)
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radius, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)

        avg_collision_rate = torch.count_nonzero(has_collision) / B
        return {
            "avg_target_error": avg_target_error,
            "avg_collision_rate": avg_collision_rate,
        }

    def validation_step_end(  # type: ignore[override]
        self, batch_parts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            "avg_target_error": torch.mean(batch_parts["avg_target_error"]),
            "avg_collision_rate": torch.mean(batch_parts["avg_collision_rate"]),
        }

    def validation_epoch_end(  # type: ignore[override]
        self, validation_step_outputs: Sequence[Dict[str, torch.Tensor]]
    ):
        avg_target_error = torch.mean(
            torch.stack([x["avg_target_error"] for x in validation_step_outputs])
        )
        self.log("avg_target_error", avg_target_error)

        avg_collision_rate = torch.mean(
            torch.stack([x["avg_collision_rate"] for x in validation_step_outputs])
        )
        self.log("avg_collision_rate", avg_collision_rate)


class MPiNetsPointNet(pl.LightningModule):
    # No changes needed here, but included for completeness of the file
    def __init__(self):
        super().__init__()
        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()

        # 1st SA: 128 points, radius .05, 64 neighbours, [1→64→64→64]
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.05,
                nsample=64,
                mlp=[1, 64, 64, 64],
                bn=False,
            )
        )

        # 2nd SA: 64 points, radius .3, 64 neighbours, [64→128→128→256]
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.3,
                nsample=64,
                mlp=[64, 128, 128, 256],
                bn=False,
            )
        )

        # 3rd SA (global): no npoint, 64 neighbours, [256→512→512]
        self.SA_modules.append(
            PointnetSAModule(
                nsample=64,
                mlp=[256, 512, 512],
                bn=False,
            )
        )

        # FC head: 512→2048→1024→1024 with GroupNorm + LeakyReLU
        self.fc_layer = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GroupNorm(16, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.GroupNorm(16, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1024),
        )

    @staticmethod
    def _break_up_pc(pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        assert point_cloud.size(2) == 4
        xyz, features = self._break_up_pc(point_cloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.fc_layer(features.squeeze(-1))