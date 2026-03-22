import torch
from torch import nn
from utils import FrankaSampler  # In order to compute self-collision loss
from robofin.pointcloud.torch import FrankaCollisionSampler
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule

from mpinets import loss
from mpinets.utils import unnormalize_franka_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Tuple, Sequence, Dict, Callable
from mpinets.model import MPiNetsPointNet
from robofin.robots import FrankaRealRobot
from loss import compute_pose_loss_rotmat, collision_loss
import torch.utils.checkpoint as checkpoint

ROLLOUT_LENGTH = 69  # The trajectory length will be ROLLOUT_LENGTH + 1

# Self-collision pairs for Franka Emika Panda robot
FRANKA_SELF_COLLISION_PAIRS = [
    ("panda_link0", "panda_link2"),
    ("panda_link0", "panda_link3"),
    ("panda_link0", "panda_link4"),
    ("panda_link0", "panda_link5"),
    ("panda_link0", "panda_link6"),
    ("panda_link0", "panda_link7"),
    ("panda_link0", "panda_hand"),
    ("panda_link1", "panda_link3"),
    ("panda_link1", "panda_link4"),
    ("panda_link1", "panda_link5"),
    ("panda_link1", "panda_link6"),
    ("panda_link1", "panda_link7"),
    ("panda_link1", "panda_hand"),
    ("panda_link2", "panda_link4"),
    ("panda_link2", "panda_link5"),
    ("panda_link2", "panda_link6"),
    ("panda_link2", "panda_link7"),
    ("panda_link2", "panda_hand"),
    ("panda_link3", "panda_link5"),
    ("panda_link3", "panda_link6"),
    ("panda_link3", "panda_link7"),
    ("panda_link3", "panda_hand"),
    ("panda_link4", "panda_link6"),
    ("panda_link4", "panda_link7"),
    ("panda_link4", "panda_hand"),
    ("panda_link5", "panda_hand"),
    # Note: 'panda_hand' in the robofin URDF includes the gripper fingers.
]


class MotionPolicyNetwork(pl.LightningModule):
    """
    The architecture laid out here is the default architecture laid out in the
    Motion Policy Networks paper (Fishman, et. al, 2022).
    """

    def __init__(self):
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
        self.decoder = nn.Sequential(
            nn.Linear(1024 + 64 + 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 7),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(
        self, xyz: torch.Tensor, q: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.fix_point_cloud_encoder:
            pc_encoding = self.point_cloud_encoder(xyz)
        else:
            pc_encoding = checkpoint.checkpoint(self.point_cloud_encoder, xyz)
        config_encoding = self.config_encoder(q)
        target_encoding = self.target_encoder(target)
        x = torch.cat((pc_encoding, config_encoding, target_encoding), dim=1)
        return self.decoder(x)


class TrainingPolicyNetOpt(MotionPolicyNetwork):
    def __init__(
        self,
        num_robot_points: int,
        goal_loss_weight: float,
        collision_loss_weight: float,
        self_collision_loss_weight: float = 2.0,
        use_self_collision: bool = False,
        smoothness_weight: float = 0.1,
        use_smoothness_loss: bool = True,
        eff_pos_path_weight: float = 0.02,
        use_eff_pos_path_loss: bool = True,
        eff_orient_path_weight: float = 0.1,
        use_eff_orient_path_loss: bool = False,
        fix_point_cloud_encoder: bool = False,
    ):
        super().__init__()
        self.num_robot_points = num_robot_points
        self.fk_sampler = None
        self.collision_sampler = None
        self.goal_loss_weight = goal_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.self_collision_loss_weight = self_collision_loss_weight
        self.use_self_collision = use_self_collision
        self.smoothness_weight = smoothness_weight
        self.use_smoothness_loss = use_smoothness_loss
        self.eff_pos_path_weight = eff_pos_path_weight
        self.use_eff_pos_path_loss = use_eff_pos_path_loss
        self.eff_orient_path_weight = eff_orient_path_weight
        self.use_eff_orient_path_loss = use_eff_orient_path_loss
        self.fix_point_cloud_encoder = fix_point_cloud_encoder
        self.validation_step_outputs = []

        # Update the point cloud encoder or not
        if self.fix_point_cloud_encoder:
            for params in self.point_cloud_encoder.parameters():
                params.requires_grad = False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
        """
        PyTorch Lightning hook: clip gradients before optimizer step
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2.0)

    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        unnormalize: bool = True,
    ) -> List[torch.Tensor]:

        xyz, q, target_pose = (
            batch["xyz"],
            batch["configuration"],
            batch["target_pose"],
        )

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
            q = torch.clamp(q + self(xyz, q, target_pose), min=-1, max=1)
            q_unnorm = unnormalize_franka_joints(q)
            assert isinstance(q_unnorm, torch.Tensor)
            q_unnorm = q_unnorm.type_as(q)
            if unnormalize:
                trajectory.append(q_unnorm)
            else:
                trajectory.append(q)

            with torch.no_grad():
                samples = sampler(q_unnorm).type_as(xyz)
                xyz[:, : samples.shape[1], :3] = samples

        return trajectory

    def eval_rollout(
        self, rollout: List[torch.Tensor], batch: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Enhanced with Action Smoothness Loss
        """
        assert len(rollout) > 0, "Rollout must contain at least one configuration"

        # Goal reaching loss
        last_configuration = rollout[-1]
        pred_pose = self.fk_sampler.end_effector_pose(last_configuration)
        target_pose = batch["target_pose"]

        position_loss, rotation_loss = compute_pose_loss_rotmat(pred_pose, target_pose)
        goal_loss = torch.mean(position_loss + 0.1 * rotation_loss)

        # Collision loss over the entire rollout
        (
            cuboid_centers,
            cuboid_dims,
            cuboid_quats,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quats,
        ) = (
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )

        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(self.device, use_cache=True)

        # Sum collision loss for each configuration in the rollout
        total_colli_loss = 0.0
        total_self_colli_loss = 0.0

        for q in rollout:
            # Environment collision loss
            input_pc = self.fk_sampler.sample(q, self.num_robot_points)
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
            total_colli_loss += colli_loss

            # Self-collision loss (only if enabled)
            if self.use_self_collision:
                per_link_pcs = self.fk_sampler.sample_per_link(
                    q, total_points=self.num_robot_points
                )

                for link_a_name, link_b_name in FRANKA_SELF_COLLISION_PAIRS:
                    if link_a_name in per_link_pcs and link_b_name in per_link_pcs:
                        pc_a = per_link_pcs[link_a_name]
                        pc_b = per_link_pcs[link_b_name]

                        dists = torch.cdist(pc_a, pc_b)
                        min_dists = torch.min(dists.view(dists.shape[0], -1), dim=1)[0]
                        self_colli_loss = torch.sum(
                            torch.clamp(0.03 - min_dists, min=0)
                        )
                        total_self_colli_loss += self_colli_loss

        # ACTION SMOOTHNESS LOSS - Core addition
        smoothness_loss = 0.0
        if self.use_smoothness_loss and len(rollout) > 1:
            # Compute joint velocities between consecutive configurations
            for i in range(len(rollout) - 1):
                # Velocity: difference between consecutive configurations
                velocity = rollout[i + 1] - rollout[i]

                # L2 norm of velocity (encourages small, smooth changes)
                velocity_penalty = torch.mean(torch.sum(velocity**2, dim=1))
                smoothness_loss += velocity_penalty

                # Optional: Add acceleration penalty for even smoother motions
                if i < len(rollout) - 2:
                    acceleration = rollout[i + 2] - 2 * rollout[i + 1] + rollout[i]
                    acceleration_penalty = torch.mean(torch.sum(acceleration**2, dim=1))
                    smoothness_loss += (
                        0.3 * acceleration_penalty
                    )  # Lower weight for acceleration

            # Normalize by number of transitions
            smoothness_loss = smoothness_loss / (len(rollout) - 1)

        # END EFFECTOR PATH LENGTH LOSSES
        eff_pos_path_loss = 0.0
        eff_orient_path_loss = 0.0
        if (self.use_eff_pos_path_loss or self.use_eff_orient_path_loss) and len(
            rollout
        ) > 1:
            eff_poses = [self.fk_sampler.end_effector_pose(q) for q in rollout]

            # Position path length loss
            if self.use_eff_pos_path_loss:
                eff_positions = torch.stack(
                    [pose[:, :3, -1] for pose in eff_poses], dim=1
                )
                pos_diffs = torch.linalg.norm(torch.diff(eff_positions, dim=1), dim=-1)
                eff_pos_path_loss = torch.mean(torch.sum(pos_diffs, dim=1))

            # Orientation path length loss
            if self.use_eff_orient_path_loss:
                for i in range(len(eff_poses) - 1):
                    R1 = eff_poses[i][:, :3, :3]
                    R2 = eff_poses[i + 1][:, :3, :3]
                    trace = torch.einsum("bii->b", torch.bmm(R1.transpose(1, 2), R2))
                    angle_diff = torch.acos(
                        torch.clamp((trace - 1) / 2, min=-1 + 1e-6, max=1 - 1e-6)
                    )
                    eff_orient_path_loss += torch.mean(
                        torch.abs(torch.rad2deg(angle_diff))
                    )

                # Normalize by number of transitions
                eff_orient_path_loss = eff_orient_path_loss / (len(rollout) - 1)

        # Calculate total loss with smoothness and path length terms
        if self.use_self_collision:
            train_loss = (
                self.goal_loss_weight * goal_loss
                + self.collision_loss_weight * total_colli_loss
                + self.self_collision_loss_weight * total_self_colli_loss
                + self.smoothness_weight * smoothness_loss
                + self.eff_pos_path_weight * eff_pos_path_loss
                + self.eff_orient_path_weight * eff_orient_path_loss
            )
        else:
            train_loss = (
                self.goal_loss_weight * goal_loss
                + self.collision_loss_weight * total_colli_loss
                + self.smoothness_weight * smoothness_loss
                + self.eff_pos_path_weight * eff_pos_path_loss
                + self.eff_orient_path_weight * eff_orient_path_loss
            )

        return (
            train_loss,
            goal_loss,
            total_colli_loss,
            total_self_colli_loss,
            torch.mean(position_loss),
            torch.mean(rotation_loss),
            smoothness_loss,  # Return for logging
            eff_pos_path_loss,  # Return for logging
            eff_orient_path_loss,  # Return for logging
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Updated to log smoothness loss
        """
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        if self.collision_sampler is None:
            self.collision_sampler = FrankaCollisionSampler(
                self.device, with_base_link=False
            )
        rollout = self.rollout(batch, ROLLOUT_LENGTH, self.sample)

        # Unpack all losses including smoothness_loss and path length losses
        (
            train_loss,
            goal_loss,
            colli_loss,
            self_colli_loss,
            position_loss,
            rotation_loss,
            smoothness_loss,
            eff_pos_path_loss,
            eff_orient_path_loss,
        ) = self.eval_rollout(rollout, batch)

        # Log all losses
        self.log("train_loss", train_loss)
        self.log("goal_loss", goal_loss)
        self.log("colli_loss", colli_loss)
        if self.use_self_collision:
            self.log("self_colli_loss", self_colli_loss)
        self.log("position_loss", position_loss)
        self.log("rotation_loss", rotation_loss)
        if self.use_smoothness_loss:
            self.log("smoothness_loss", smoothness_loss)
        if self.use_eff_pos_path_loss:
            self.log("eff_pos_path_loss", eff_pos_path_loss)
        if self.use_eff_orient_path_loss:
            self.log("eff_orient_path_loss", eff_orient_path_loss)

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

        with torch.no_grad():
            if self.fk_sampler is None:
                self.fk_sampler = FrankaSampler(self.device, use_cache=True)
            if self.collision_sampler is None:
                self.collision_sampler = FrankaCollisionSampler(
                    self.device, with_base_link=False
                )
            rollout = self.rollout(batch, ROLLOUT_LENGTH, self.sample)

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
            assert rollout.shape == (B, ROLLOUT_LENGTH + 1, 7)
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
                assert sdf_values.shape == (B, ROLLOUT_LENGTH + 1, num_spheres)
                radius_collisions = torch.any(
                    sdf_values.reshape((sdf_values.size(0), -1)) <= radius, dim=-1
                )
                has_collision = torch.logical_or(radius_collisions, has_collision)

            avg_collision_rate = torch.count_nonzero(has_collision) / B

            result = {
                "avg_target_error": avg_target_error,
                "avg_collision_rate": avg_collision_rate,
            }
            self.validation_step_outputs.append(result)
            return result

    def on_validation_epoch_end(self):
        avg_target_error = torch.mean(
            torch.stack([x["avg_target_error"] for x in self.validation_step_outputs])
        )
        self.log("avg_target_error", avg_target_error)

        avg_collision_rate = torch.mean(
            torch.stack([x["avg_collision_rate"] for x in self.validation_step_outputs])
        )
        self.log("avg_collision_rate", avg_collision_rate)

        self.validation_step_outputs.clear()
