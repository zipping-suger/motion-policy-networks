import torch
from torch import nn
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from mpinets.utils import unnormalize_franka_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Tuple, Sequence, Dict, Callable, Optional


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=device))
            * torch.arange(0, half_dim, device=device)
            / half_dim
        )
        args = x.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class FlowMatchingMotionPolicyNetwork(pl.LightningModule):
    """
    Motion Policy Network modified for flow matching training
    """

    def __init__(
        self,
        flow_sig_min: float = 0.001,
        num_inference_steps: int = 10,
        final_action_clip_value: Optional[float] = None,
        learning_rate: float = 1e-4,
        num_robot_points: int = 1024,
        flow_sampling: str = "beta",  # NEW: Add flow_sampling parameter
        flow_alpha: float = 1.5,  # NEW: Beta distribution parameters
        flow_beta: float = 1.0,  # NEW: Beta distribution parameters
    ):
        super().__init__()
        self.save_hyperparameters()

        # NEW: Initialize beta distribution for sampling
        if flow_sampling == "beta":
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
            self.flow_t_max = 1 - flow_sig_min

        self.register_buffer(
            "displacement_std",
            torch.tensor([0.01] * 7),
        )
        
        # Original components remain the same...
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
            nn.Linear(12, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
        )

        self.time_embedding = SinusoidalPosEmb(64, max_period=10000)
        self.time_proj = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024 + 64 + 64 + 64 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 7),
        )

        self.num_robot_points = num_robot_points
        self.fk_sampler = None
        self.collision_sampler = None
        
    def configure_optimizers(self):
        """
        Configure the optimizer for training
        """
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )
        return optimizer

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        """
        Sample flow matching timesteps using the specified strategy

        Args:
            bsz: batch size

        Returns:
            t: timesteps of shape (bsz,)
        """
        if self.hparams.flow_sampling == "uniform":
            # Stratified uniform sampling (like the example code)
            eps = 1e-5
            t = (
                torch.rand(1, device=self.device)
                + torch.arange(bsz, device=self.device) / bsz
            ) % (1 - eps)
        elif self.hparams.flow_sampling == "beta":
            # Beta distribution sampling (like the example code)
            z = self.flow_beta_dist.sample((bsz,)).to(self.device)
            t = self.flow_t_max * (1 - z)  # flip and shift
        else:
            # Simple uniform (your original implementation)
            t = torch.rand(bsz, device=self.device)

        return t

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow - interpolates between noise and target"""
        t = t[:, None]  # (B, 1) for broadcasting
        return (1 - (1 - self.hparams.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        xyz: torch.Tensor,
        q: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noisy_action: torch.Tensor,
    ) -> torch.Tensor:
        # ... rest of forward remains the same ...
        pc_encoding = self.point_cloud_encoder(xyz)
        config_encoding = self.config_encoder(q)
        target_encoding = self.target_encoder(target)
        action_encoding = self.action_encoder(noisy_action)
        time_emb = self.time_embedding(t)
        time_encoding = self.time_proj(time_emb)

        x = torch.cat(
            (
                pc_encoding,
                config_encoding,
                target_encoding,
                time_encoding,
                action_encoding,
            ),
            dim=1,
        )
        return self.decoder(x)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Flow matching training step - MODIFIED to use sample_fm_time
        """
        xyz, q, target_pose, next_q = (
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
            batch["supervision"],
        )

        # Calculate target displacement
        target_displacement = next_q - q

        # MODIFIED: Use sample_fm_time instead of simple uniform
        t = self.sample_fm_time(xyz.size(0))  # This is the key change!

        x0 = self.displacement_std * torch.randn_like(target_displacement)
        psi_t = self.psi_t(x0, target_displacement, t)
        v_pred = self(xyz, q, target_pose, t, psi_t)
        d_psi = target_displacement - (1 - self.hparams.flow_sig_min) * x0

        loss = torch.mean((v_pred - d_psi) ** 2)
        self.log("train_loss", loss)
        return loss

    def infer_action(
        self,
        xyz: torch.Tensor,
        q: torch.Tensor,
        target: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference using Euler integration
        """
        if num_steps is None:
            num_steps = self.hparams.num_inference_steps

        # Start with noise
        action = self.displacement_std * torch.randn_like(q)  # Scaled!

        delta_t = 1.0 / num_steps
        t = torch.zeros(xyz.size(0), device=self.device)

        for _ in range(num_steps):
            # Predict velocity at current time and noisy action
            velocity = self(xyz, q, target, t, action)

            # Euler step
            action = action + delta_t * velocity
            t = t + delta_t

        # Clamp if specified
        if self.hparams.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.hparams.final_action_clip_value,
                self.hparams.final_action_clip_value,
            )

        return action

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        unnormalize: bool = False,
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
        :rtype List[torch.Tensor]: The entire trajectory batch, i.e. a list of
                                   configuration batches including the starting
                                   configurations where each element in the list
                                   corresponds to a timestep.
        """
        xyz, q, target_pose = (
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
        )
        # This block is to adapt for the case where we only want to roll out a
        # single trajectory
        if q.ndim == 1:
            xyz = xyz.unsqueeze(0)
            q = q.unsqueeze(0)
        if unnormalize:
            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            trajectory = [q_unnorm]
        else:
            trajectory = [q]

        for i in range(rollout_length):
            # Use flow matching inference instead of direct forward pass
            displacement = self.infer_action(xyz, q, target_pose)
            q = torch.clamp(q + displacement, min=-1, max=1)

            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            q_unnorm = q_unnorm.type_as(q)
            if unnormalize:
                trajectory.append(q_unnorm)
            else:
                trajectory.append(q)

            samples = sampler(q_unnorm).type_as(xyz)
            xyz[:, : samples.shape[1], :3] = samples

        return trajectory

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q, self.num_robot_points)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation with rollout and collision checking
        """
        # Initialize samplers if needed
        if self.fk_sampler is None:
            from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler

            self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        if self.collision_sampler is None:
            from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler

            self.collision_sampler = FrankaCollisionSampler(
                self.device, with_base_link=False
            )

        # Perform rollout
        rollout = self.rollout(batch, 69, self.sample, unnormalize=True)

        # Calculate target error
        assert self.fk_sampler is not None
        eff = self.fk_sampler.end_effector_pose(rollout[-1])
        position_error = torch.linalg.vector_norm(
            eff[:, :3, -1] - batch["target_position"], dim=1
        )
        avg_target_error = torch.mean(position_error)

        # Calculate collision rate
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
        rollout_tensor = torch.stack(rollout, dim=1)

        # Check for collisions in the rollout
        assert rollout_tensor.shape == (B, 70, 7)
        rollout_flat = rollout_tensor.reshape(-1, 7)
        has_collision = torch.zeros(B, dtype=torch.bool, device=self.device)

        collision_spheres = self.collision_sampler.compute_spheres(rollout_flat)
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

    def validation_step_end(
        self, batch_parts: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Called by Pytorch Lightning at the end of each validation step to
        aggregate across devices
        """
        return {
            "avg_target_error": torch.mean(batch_parts["avg_target_error"]),
            "avg_collision_rate": torch.mean(batch_parts["avg_collision_rate"]),
        }

    def validation_epoch_end(
        self, validation_step_outputs: Sequence[Dict[str, torch.Tensor]]
    ):
        """
        Pytorch lightning method that aggregates stats from the validation loop and logs
        """
        avg_target_error = torch.mean(
            torch.stack([x["avg_target_error"] for x in validation_step_outputs])
        )
        self.log("val_avg_target_error", avg_target_error)

        avg_collision_rate = torch.mean(
            torch.stack([x["avg_collision_rate"] for x in validation_step_outputs])
        )
        self.log("val_avg_collision_rate", avg_collision_rate)


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
