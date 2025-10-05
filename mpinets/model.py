import torch
from torch import nn
from typing import Optional, Tuple
import numpy as np
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule

from mpinets import loss
from mpinets.utils import unnormalize_franka_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Tuple, Sequence, Dict, Callable


from torch_geometric.nn import MLP, PointNetConv, fps, radius
from torch_geometric.utils import to_dense_batch

from mpinets.transformer import (
    Encoder,
    FeedForward,
    MultiHeadAttention,
    TransformerLayer,
)


class PositionEncoding3D(pl.LightningModule):
    """
    Generate sinusoidal positional encoding.

    f(p) = (sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^L pi pi), cos(2^L pi p))
    From M2t2:
    https://github.com/NVlabs/M2T2/blob/734a5251e7ca36405c2b7056407db90db6c8e695/m2t2/contact_decoder.py#L51
    The primary difference from the source is that there was
    a bug in the positional encoding, which is fixed here.
    """

    def __init__(self, enc_dim, scale=np.pi, temperature=10000):
        super(PositionEncoding3D, self).__init__()
        self.enc_dim = enc_dim
        self.freq = np.ceil(enc_dim / 6)
        self.scale = scale
        self.temperature = temperature
        # Register pc_bounds as a buffer so it moves with the model
        self.register_buffer(
            "pc_bounds", torch.tensor([[-1.5, -1.5, -0.1], [1.5, 1.5, 1.5]])
        )

    def forward(self, pos, bounds=None):
        # Use stored bounds if none provided, otherwise use the provided ones
        if bounds is None:
            bounds = self.pc_bounds
        else:
            # Ensure bounds are on the same device as pos
            bounds = bounds.to(pos.device)

        pos_min = bounds[0]
        pos_max = bounds[1]
        pos = ((pos - pos_min) / (pos_max - pos_min) - 0.5) * 2 * np.pi
        dim_t = torch.arange(self.freq, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (dim_t / self.freq)
        pos = pos[..., None] * self.scale / dim_t  # (B, N, 3, F)
        pos = torch.stack([pos.sin(), pos.cos()], dim=-1).flatten(start_dim=2)
        pos = pos[..., : self.enc_dim]
        return pos.detach()


class SAModule(pl.LightningModule):
    """
    Set aggregation module from PointNet++ (based on implementation in pytorch geometric).
    """

    def __init__(self, ratio: float, r: float, net: nn.Module):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(net, add_self_loops=False)

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class MPiFormerPointNet(pl.LightningModule):
    def __init__(self, num_robot_points: int, input_feature_dim: int, d_model: int):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            ratio=0.25, r=0.05, net=MLP([3 + input_feature_dim, 64, 64, 64])
        )
        self.sa2_module = SAModule(ratio=0.25, r=0.3, net=MLP([64 + 3, 128, 128, 256]))
        self.sa3_module = SAModule(
            ratio=0.25, r=0.5, net=MLP([256 + 3, 256, 512, d_model])
        )
        self.point_id_embedding = nn.Parameter(
            torch.randn((1, num_robot_points, input_feature_dim))
        )
        self.feature_encoder = nn.Embedding(3, input_feature_dim)
        self.num_robot_points = num_robot_points

    def forward(
        self,
        point_cloud_features: torch.Tensor,
        point_cloud: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass of the network

        :param point_cloud torch.Tensor: Has dimensions (B, N, 4)
                                              B is the batch size
                                              N is the number of points
                                              4 is x, y, z, segmentation_mask
                                              This tensor must be on the GPU (CPU tensors not supported)
        :rtype torch.Tensor: The output from the network
        """
        B, N, _ = point_cloud.shape
        pos = point_cloud.reshape(B * N, 3)  # Hard coded to fail if dimensions change
        x = self.feature_encoder(point_cloud_features.squeeze(-1).long())
        robot_features = x[:, : self.num_robot_points, :]
        other_features = x[:, self.num_robot_points:, :]
        robot_features = robot_features + self.point_id_embedding
        x = torch.cat((robot_features, other_features), dim=1)
        x = x.reshape(B * N, -1)

        batch_indices = torch.arange(B, device=point_cloud.device).unsqueeze(1)
        batch_indices = batch_indices.repeat(1, N)
        batch_indices = batch_indices.view(-1)

        x, pos, batch_indices = self.sa1_module(x, pos, batch_indices)
        x, pos, batch_indices = self.sa2_module(x, pos, batch_indices)
        x, pos, batch_indices = self.sa3_module(x, pos, batch_indices)
        x, x_mask = to_dense_batch(x, batch_indices)
        assert torch.all(x_mask), "Should be true because this PC has consistent size"
        pos, pos_mask = to_dense_batch(pos, batch_indices)
        assert torch.all(pos_mask), "Should be true because this PC has consistent size"
        return x, pos


class MotionPolicyTransformer(pl.LightningModule):
    def __init__(
        self,
        num_robot_points: int,
        *,
        feature_dim: int = 4,
        n_heads: int = 8,
        d_model: int = 512,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.point_cloud_embedder = MPiFormerPointNet(
            num_robot_points, feature_dim, d_model
        )
        self.feature_embedder = nn.Linear(7, d_model)
        self.action_decoder = nn.Linear(d_model, 7)
        encoder_layer = TransformerLayer(
            d_model=d_model,
            self_attn=MultiHeadAttention(
                heads=n_heads,
                d_model=d_model,
                dropout_prob=dropout,
            ),
            src_attn=None,  # No cross attention
            feed_forward=FeedForward(
                d_model=d_model,
                d_ff=4 * d_model,
                dropout=dropout,
                activation=nn.GELU,  # GELU gating
                is_gated=False,  # GELU gating
                bias1=True,  # GELU gating
                bias2=True,  # GELU gating
                bias_gate=True,  # GELU gating
            ),
            dropout_prob=dropout,
        )
        self.encoder = Encoder(encoder_layer, n_layers=n_layers)
        self.action_tokens = nn.Parameter(torch.randn((1, 1, d_model)))
        # Embedding instead of nn.parameter because it does gaussian initialization
        self.token_type_embedding = nn.Embedding(3, d_model)
        self.pe_layer = PositionEncoding3D(d_model)
        # Remove this line since pc_bounds is now handled inside PositionEncoding3D
        # self.pc_bounds = torch.as_tensor([[-1.5, -1.5, -0.1], [1.5, 1.5, 1.5]])

    def forward(
        self,
        *,
        point_cloud_labels: torch.Tensor,
        point_cloud: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        assert point_cloud_labels.shape[:2] == point_cloud.shape[:2]
        pc_embedding, pos = self.point_cloud_embedder(point_cloud_labels, point_cloud)
        feature_embedding = self.feature_embedder(q).unsqueeze(1)
        B = point_cloud.size(0)
        sequence = torch.cat(
            (
                pc_embedding,
                feature_embedding,
                self.action_tokens.expand((B, -1, -1)),
            ),
            dim=1,
        ).transpose(0, 1)

        # Indicator embeddings to label the token type
        pc_type_emb = self.token_type_embedding(
            torch.tensor(0, dtype=torch.long, device=self.device)
        )
        joint_state_type_emb = self.token_type_embedding(
            torch.tensor(1, dtype=torch.long, device=self.device)
        )[None, None, :]
        action_type_emb = self.token_type_embedding(
            torch.tensor(2, dtype=torch.long, device=self.device)
        )[None, None, :]

        pos_emb = torch.cat(
            (
                # Use both sin/cos emb and type label emb for pc
                self.pe_layer(pos) + pc_type_emb,  # Remove pc_bounds argument
                joint_state_type_emb.expand((B, -1, -1)),
                action_type_emb.expand((B, 1, -1)),
            ),
            dim=1,
        ).transpose(0, 1)
        embedded_sequence = sequence + pos_emb
        action = self.encoder(embedded_sequence, mask=None)[-1:]
        return self.action_decoder(action).transpose(0, 1)


class TrainingMotionPolicyNetwork(MotionPolicyTransformer):
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
    ):
        """
        Creates the network and assigns additional parameters for training


        :param num_robot_points int: The number of robot points used when resampling
                                     the robot points during rollouts (used in validation)
        :param point_match_loss_weight float: The weight assigned to the behavior
                                              cloning loss.
        :param collision_loss_weight float: The weight assigned to the collision loss
        :rtype Self: An instance of the network
        """
        super().__init__(num_robot_points=num_robot_points)
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.fk_sampler = None
        self.collision_sampler = None
        self.loss_fun = loss.CollisionAndBCLossContainer()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

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
                                   corresponds to a timestep. For example, the
                                   first element of each batch in the list would
                                   be a single trajectory.
        """

        point_cloud_labels, xyz, q, target_pose = (
            batch["point_cloud_labels"],
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
            qdelta = self(point_cloud_labels=point_cloud_labels, point_cloud=xyz, q=q).squeeze()
            q = torch.clamp(
                q + qdelta,
                min=-1,
                max=1,
            )
            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            q_unnorm = q_unnorm.type_as(q)
            if unnormalize:
                trajectory.append(q_unnorm)
            else:
                trajectory.append(q)

            # Fix: Ensure q_unnorm has the correct dimensions for sampling
            if q_unnorm.ndim == 3:
                samples = sampler(q_unnorm.squeeze(1)).type_as(xyz)
            else:
                samples = sampler(q_unnorm).type_as(xyz)
            xyz[:, : samples.shape[1], :3] = samples

        return trajectory

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
        point_cloud_labels, xyz, q, target_pose = (
            batch["point_cloud_labels"],
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
        )
        y_hat = torch.clamp(
            q + self(point_cloud_labels=point_cloud_labels, point_cloud=xyz, q=q).squeeze(),
            min=-1,
            max=1,
        )

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
        collision_loss, point_match_loss = self.loss_fun(
            y_hat,
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
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

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q, self.num_robot_points)

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        This is a Pytorch Lightning function run automatically across devices
        during the validation loop

        :param batch Dict[str, torch.Tensor]: The batch coming from the dataloader
        :param batch_idx int: The index of the batch (not used by this function)
        :rtype torch.Tensor: The loss values which are to be collected into summary stats
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
        """
        Called by Pytorch Lightning at the end of each validation step to
        aggregate across devices

        :param batch_parts Dict[str, torch.Tensor]: The parts accumulated from all devices
        :rtype Dict[str, torch.Tensor]: The average values across the devices
        """
        return {
            "avg_target_error": torch.mean(batch_parts["avg_target_error"]),
            "avg_collision_rate": torch.mean(batch_parts["avg_collision_rate"]),
        }

    def validation_epoch_end(  # type: ignore[override]
        self, validation_step_outputs: Sequence[Dict[str, torch.Tensor]]
    ):
        """
        Pytorch lightning method that aggregates stats from the validation loop and logs

        :param validation_step_outputs Sequence[Dict[str, torch.Tensor]]: The outputs from each
                                                                      validation step
        """
        avg_target_error = torch.mean(
            torch.stack([x["avg_target_error"] for x in validation_step_outputs])
        )
        self.log("avg_target_error", avg_target_error)

        avg_collision_rate = torch.mean(
            torch.stack([x["avg_collision_rate"] for x in validation_step_outputs])
        )
        self.log("avg_collision_rate", avg_collision_rate)
