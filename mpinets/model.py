import torch
from torch import nn
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from mpinets import loss
from mpinets.utils import unnormalize_franka_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Tuple, Sequence, Dict, Callable, Optional


class MotionPolicyNetwork(pl.LightningModule):
    """
    The architecture laid out here is the default architecture laid out in the
    Motion Policy Networks paper (Fishman, et. al, 2022).
    """

    def __init__(
        self,
        action_chunk_length: int = 1,
        min_lr: float = 1e-6,  # NEW: Learning rate scheduling
        max_lr: float = 1e-4,
        warmup_steps: int = 1000,
        decay_rate: float = 0.95,
    ):
        """
        Constructs the model

        :param action_chunk_length int: Number of consecutive actions to predict (default: 1)
        :param min_lr float: Minimum learning rate
        :param max_lr float: Maximum learning rate
        :param warmup_steps int: Number of warmup steps
        :param decay_rate float: Learning rate decay rate
        """
        super().__init__()
        self.action_chunk_length = action_chunk_length
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate

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
        # NEW: Update decoder to output action_chunk_length * 7 dimensions
        self.decoder = nn.Sequential(
            nn.Linear(1024 + 64 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.action_chunk_length * 7),  # Output multiple actions
        )

    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer with learning rate scheduling
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.min_lr, weight_decay=1e-4, betas=(0.9, 0.95)
        )

        # Lambda function for the linear warmup and decay
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                lr = self.min_lr + (self.max_lr - self.min_lr) * min(
                    1.0, step / self.warmup_steps
                )
            else:
                # Exponential decay after warmup
                decay_steps = step - self.warmup_steps
                lr = self.max_lr * (self.decay_rate ** (decay_steps / 1000))
                lr = max(lr, self.min_lr)
            return lr / self.min_lr

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(
        self, xyz: torch.Tensor, q: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Passes data through the network to produce an output

        :param xyz torch.Tensor: Tensor representing the point cloud. Should
                                      have dimensions of [B x N x 4] where B is the batch
                                      size, N is the number of points and 4 is because there
                                      are three geometric dimensions and a segmentation mask
        :param q torch.Tensor: The current robot configuration normalized to be between
                                    -1 and 1, according to each joint's range of motion
        :rtype torch.Tensor: The displacement to be applied to the current configuration to get
                     the position at the next step (still in normalized space)
        """
        pc_encoding = self.point_cloud_encoder(xyz)
        config_encoding = self.config_encoder(q)
        target_encoding = self.target_encoder(target)
        x = torch.cat((pc_encoding, config_encoding, target_encoding), dim=1)
        # NEW: Reshape output to [batch_size, action_chunk_length, 7]
        output = self.decoder(x)
        return output.view(-1, self.action_chunk_length, 7)


class TrainingMotionPolicyNetwork(MotionPolicyNetwork):
    """
    An version of the MotionPolicyNetwork model that has additional attributes
    necessary during training (or using the validation step outside of the
    training process). This class is a valid model, but it's overkill when
    doing real robot inference and, for example, point cloud sampling is
    done by an outside process (such as downsampling point clouds from a point cloud).
    """

    def __init__(
        self,
        num_robot_points: int,
        point_match_loss_weight: float,
        collision_loss_weight: float,
        action_chunk_length: int = 1,
        collision_loss_margin: float = 0.03,  # NEW: Configurable margin
        min_lr: float = 1e-6,
        max_lr: float = 1e-4,
        warmup_steps: int = 1000,
        decay_rate: float = 0.95,
    ):
        """
        Creates the network and assigns additional parameters for training

        :param num_robot_points int: The number of robot points used when resampling
                                     the robot points during rollouts (used in validation)
        :param point_match_loss_weight float: The weight assigned to the behavior
                                              cloning loss.
        :param collision_loss_weight float: The weight assigned to the collision loss
        :param action_chunk_length int: Number of consecutive actions to predict (default: 1)
        :param collision_loss_margin float: Margin for collision loss (default: 0.03)
        :param min_lr float: Minimum learning rate
        :param max_lr float: Maximum learning rate
        :param warmup_steps int: Number of warmup steps
        :param decay_rate float: Learning rate decay rate
        :rtype Self: An instance of the network
        """
        super().__init__(
            action_chunk_length=action_chunk_length,
            min_lr=min_lr,
            max_lr=max_lr,
            warmup_steps=warmup_steps,
            decay_rate=decay_rate,
        )
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.collision_loss_margin = collision_loss_margin  # NEW: Store margin

        self.fk_sampler = None
        self.collision_sampler = None
        self.loss_fun = loss.CollisionAndBCLossContainer(
            collision_loss_margin=collision_loss_margin  # NEW: Pass margin to loss container
        )

        # NEW: Add metrics for better validation tracking
        self.val_position_error = torchmetrics.MeanMetric()
        self.val_orientation_error = torchmetrics.MeanMetric()
        self.val_collision_rate = torchmetrics.MeanMetric()
        self.val_success_rate = torchmetrics.MeanMetric()

    def setup(self, stage: Optional[str] = None):
        """
        Sets up the model by getting the device and initializing the collision and FK samplers.
        """
        device = self.device
        self.collision_sampler = FrankaCollisionSampler(device, with_base_link=False)
        self.fk_sampler = FrankaSampler(
            device,
            use_cache=True,
        )

    def state_based_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NEW: Separate method for state-based training step
        """
        xyz, q, target_pose = (
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
        )

        # Get action chunk predictions [B, action_chunk_length, 7]
        action_chunk = self(xyz, q, target_pose)

        # Compute cumulative actions to get predicted configurations
        cumulative_actions = torch.cumsum(
            action_chunk, dim=1
        )  # [B, action_chunk_length, 7]
        q_expanded = q.unsqueeze(1).expand(
            -1, self.action_chunk_length, -1
        )  # [B, action_chunk_length, 7]
        y_hats = torch.clamp(
            q_expanded + cumulative_actions, min=-1, max=1
        )  # [B, action_chunk_length, 7]

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
            batch["supervision"],  # This should now be [B, action_chunk_length, 7]
        )

        # NEW: Flatten batch and action chunk dimensions for loss computation
        # This is more efficient than looping and matches the other repo's approach
        y_hats_flat = y_hats.reshape(-1, 7)  # [B * action_chunk_length, 7]
        supervision_flat = supervision.reshape(-1, 7)  # [B * action_chunk_length, 7]

        # Repeat obstacle data for each timestep in the action chunk
        cuboid_centers_flat = cuboid_centers.repeat_interleave(
            self.action_chunk_length, dim=0
        )
        cuboid_dims_flat = cuboid_dims.repeat_interleave(
            self.action_chunk_length, dim=0
        )
        cuboid_quats_flat = cuboid_quats.repeat_interleave(
            self.action_chunk_length, dim=0
        )
        cylinder_centers_flat = cylinder_centers.repeat_interleave(
            self.action_chunk_length, dim=0
        )
        cylinder_radii_flat = cylinder_radii.repeat_interleave(
            self.action_chunk_length, dim=0
        )
        cylinder_heights_flat = cylinder_heights.repeat_interleave(
            self.action_chunk_length, dim=0
        )
        cylinder_quats_flat = cylinder_quats.repeat_interleave(
            self.action_chunk_length, dim=0
        )

        collision_loss, point_match_loss = self.loss_fun(
            y_hats_flat,
            cuboid_centers_flat,
            cuboid_dims_flat,
            cuboid_quats_flat,
            cylinder_centers_flat,
            cylinder_radii_flat,
            cylinder_heights_flat,
            cylinder_quats_flat,
            supervision_flat,
        )

        return collision_loss, point_match_loss

    def combine_training_losses(
        self, collision_loss: torch.Tensor, point_match_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        NEW: Separate method for combining losses
        """
        self.log("point_match_loss", point_match_loss)
        self.log("collision_loss", collision_loss)
        train_loss = (
            self.point_match_loss_weight * point_match_loss
            + self.collision_loss_weight * collision_loss
        )
        self.log("train_loss", train_loss)
        return train_loss

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        A function called automatically by Pytorch Lightning during training.
        This function handles the forward pass, the loss calculation, and what to log

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                                   data loader--should already be
                                                   on the correct device
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The overall weighted loss (used for backprop)
        """
        collision_loss, point_match_loss = self.state_based_step(batch)
        return self.combine_training_losses(collision_loss, point_match_loss)

    def target_error(
        self, batch: Dict[str, torch.Tensor], rollouts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        NEW: Calculate position and orientation errors between rollouts and target
        """
        assert self.fk_sampler is not None
        eff = self.fk_sampler.end_effector_pose(rollouts[:, -1])

        # Position error
        position_error = torch.linalg.vector_norm(
            eff[:, :3, -1] - batch["target_position"], dim=1
        )

        # Orientation error (similar to the other repo)
        target_rot = batch["target_rotation"].view(-1, 3, 3)  # [B, 3, 3]
        R = torch.matmul(eff[:, :3, :3], target_rot.transpose(1, 2))
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orientation_error = torch.abs(torch.rad2deg(torch.acos(cos_value)))

        return position_error, orientation_error

    def collision_error(
        self, batch: Dict[str, torch.Tensor], rollouts: torch.Tensor
    ) -> torch.Tensor:
        """
        NEW: Improved collision error calculation
        """
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

        # Reshape rollouts for collision checking
        rollout_steps = rollouts.reshape(-1, 7)
        has_collision = torch.zeros(B, dtype=torch.bool, device=self.device)

        assert self.collision_sampler is not None
        collision_spheres = self.collision_sampler.compute_spheres(rollout_steps)

        for radius, spheres in collision_spheres:
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((B, -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert sdf_values.size(0) == B and sdf_values.size(2) == num_spheres
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radius, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)

        return has_collision

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        unnormalize: bool = True,
    ) -> List[torch.Tensor]:
        """
        Rolls out the policy an arbitrary length by calling it iteratively

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                            data loader--should already be
                                            on the correct device
        :param rollout_length int: The number of steps to roll out (not including the start)
        :param sampler Callable[[torch.Tensor], torch.Tensor]: A function that takes a batch of robot
                                                            configurations [B x 7] and returns a batch of
                                                            point clouds samples on the surface of that robot
        :param unnormalize bool: Whether to return the whole trajectory unnormalized
                                (i.e. converted back into joint space)
        :rtype list[torch.Tensor]: The entire trajectory batch, i.e. a list of
                                configuration batches including the starting
                                configurations where each element in the list
                                corresponds to a timestep. For example, the
                                first element of each batch in the list would
                                be a single trajectory.
        """
        xyz, q, target_pose = (
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
        )

        B = q.size(0)

        # Calculate the number of chunks needed
        n_chunks = (
            rollout_length + self.action_chunk_length - 1
        ) // self.action_chunk_length
        actual_rollout_length = n_chunks * self.action_chunk_length

        # Initialize trajectory with starting configuration
        trajectory = [q]
        if unnormalize:
            trajectory[0] = unnormalize_franka_joints(trajectory[0])

        current_q = q
        current_xyz = xyz.clone()

        for i in range(n_chunks):
            # Get action chunk predictions [B, action_chunk_length, 7]
            action_chunk = self(current_xyz, current_q, target_pose)

            # Apply actions sequentially
            for j in range(self.action_chunk_length):
                if i * self.action_chunk_length + j >= rollout_length:
                    break

                # Update configuration
                current_q = torch.clamp(current_q + action_chunk[:, j, :], min=-1, max=1)

                # Add to trajectory
                if unnormalize:
                    trajectory.append(unnormalize_franka_joints(current_q))
                else:
                    trajectory.append(current_q.clone())

                # Update point cloud with new robot configuration
                if (
                    j < self.action_chunk_length - 1
                ):  # Don't sample after the last action in chunk
                    robot_points = sampler(current_q)
                    # Replace robot portion of point cloud (assuming first num_robot_points are robot points)
                    current_xyz[:, : self.num_robot_points, :3] = robot_points

        return trajectory

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q, self.num_robot_points)

    def state_validation_step(self, batch: Dict[str, torch.Tensor]):
        """
        NEW: Separate method for state-based validation
        """
        collision_loss, point_match_loss = self.state_based_step(batch)
        self.log("val_point_match_loss", point_match_loss)
        self.log("val_collision_loss", collision_loss)
        val_loss = (
            self.point_match_loss_weight * point_match_loss
            + self.collision_loss_weight * collision_loss
        )
        self.log("val_loss", val_loss)

    def trajectory_validation_step(self, batch: Dict[str, torch.Tensor]):
        """
        NEW: Separate method for trajectory-based validation
        """
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        if self.collision_sampler is None:
            self.collision_sampler = FrankaCollisionSampler(
                self.device, with_base_link=False
            )

        rollouts = self.rollout(batch, 69, self.sample, unnormalize=True)
        rollouts_tensor = torch.stack(rollouts, dim=1)  # [B, 70, 7]

        # Calculate errors
        position_error, orientation_error = self.target_error(batch, rollouts_tensor)
        has_collision = self.collision_error(batch, rollouts_tensor)

        # Update metrics
        self.val_position_error.update(position_error.mean())
        self.val_orientation_error.update(orientation_error.mean())
        self.val_collision_rate.update(has_collision.float().mean())

        # Calculate success rate (no collision and position error < 0.01 and orientation error < 15 degrees)
        success = torch.logical_and(
            ~has_collision,
            torch.logical_and(position_error < 0.01, orientation_error < 15),
        )
        self.val_success_rate.update(success.float().mean())

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        This is a Pytorch Lightning function run automatically across devices
        during the validation loop

        :param batch Dict[str, torch.Tensor]: The batch coming from the dataloader
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype Dict[str, torch.Tensor]: The validation metrics
        """
        # Run both state-based and trajectory-based validation
        self.state_validation_step(batch)
        self.trajectory_validation_step(batch)

        return {
            "val_position_error": self.val_position_error.compute(),
            "val_orientation_error": self.val_orientation_error.compute(),
            "val_collision_rate": self.val_collision_rate.compute(),
            "val_success_rate": self.val_success_rate.compute(),
        }

    def on_validation_epoch_end(self):
        """
        NEW: Log validation metrics at epoch end
        """
        self.log("avg_val_target_error", self.val_position_error.compute())
        self.log("avg_val_orientation_error", self.val_orientation_error.compute())
        self.log("avg_val_collision_rate", self.val_collision_rate.compute())
        self.log("avg_val_success_rate", self.val_success_rate.compute())

        # Reset metrics
        self.val_position_error.reset()
        self.val_orientation_error.reset()
        self.val_collision_rate.reset()
        self.val_success_rate.reset()


class MPiNetsPointNet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self._build_model()

    def _build_model(self):
        """
        Assembles the model design into a ModuleList
        """
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
        # features: [B, C, 1]  → squeeze → [B, C]
        return self.fc_layer(features.squeeze(-1))
