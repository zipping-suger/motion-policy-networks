import torch
from torch import nn
from robofin.pointcloud.torch import FrankaSampler, FrankaCollisionSampler
import pytorch_lightning as pl
from pointnet2_ops.pointnet2_modules import PointnetSAModule

from mpinets import loss
from mpinets.utils import unnormalize_franka_joints
from mpinets.geometry import TorchCuboids, TorchCylinders
from typing import List, Tuple, Sequence, Dict, Callable
from mpinets.model import MotionPolicyNetwork
from robofin.robots import FrankaRealRobot
from loss import compute_pose_loss_rotmat, collision_loss


ROLLOUT_LENGTH = 49  # The trajectory length will be ROLLOUT_LENGTH + 1


class TrainingPolicyNetOpt(MotionPolicyNetwork):
    def __init__(
        self,
        num_robot_points: int,
        goal_loss_weight: float,
        collision_loss_weight: float,
    ):
        """
        Creates the network and assigns additional parameters for training


        :param num_robot_points int: The number of robot points used when resampling
                                     the robot points during rollouts (used in validation)
        :rtype Self: An instance of the network
        """
        super().__init__()            
        self.num_robot_points = num_robot_points
        self.fk_sampler = None
        self.collision_sampler = None
        self.goal_loss_weight = goal_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.mse_loss = nn.MSELoss()
        self.validation_step_outputs = []
        
        # The point cloud encoder does not need to be trained
        for params in self.point_cloud_encoder.parameters():
            params.requires_grad = False
            
    def configure_optimizers(self):
        """
        A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer
        
    def rollout(
        self,
        batch: Dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
        unnormalize: bool = True,
    ) -> List[torch.Tensor]:
        
        xyz, q = (
            batch["xyz"],
            batch["configuration"],
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
            q = torch.clamp(q + self(xyz, q), min=-1, max=1)
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
    
    # Differentialble rollout evaluation function with repect to the rollout
    def eval_rollout(self, rollout: List[torch.Tensor], batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluates a rollout by computing the difference between the last configuration
        in the rollout and the target configuration, and sums collision loss over the rollout.
        :param rollout List[torch.Tensor]: A list of configurations in the rollout
        :param target_configuration torch.Tensor: The target configuration to compare against
        :rtype torch.Tensor: Scalar loss (mean squared error between last and target configuration + collision loss)
        """
        assert len(rollout) > 0, "Rollout must contain at least one configuration"

        # Goal reaching loss
        last_configuration = rollout[-1]      
        # Goal reaching loss in end-effector space
        pred_pose = self.fk_sampler.end_effector_pose(last_configuration)  # (B,4,4)
        target_pose = batch["target_pose"]  # (B,12) [x, y, z, flattened rotation matrix]

        position_loss, rotation_loss = compute_pose_loss_rotmat(
            pred_pose, target_pose
        )

        # Combine losses with balancing factor
        goal_loss = torch.mean(position_loss + 0.1 * rotation_loss) # Follow the rule of thumb, to make them roughly equal in scale
        
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
        
        for q in rollout:
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
        
        train_loss = self.goal_loss_weight * goal_loss + self.collision_loss_weight * total_colli_loss
    
        return train_loss, goal_loss, total_colli_loss, torch.mean(position_loss), torch.mean(rotation_loss)

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
        # Rollout the trajectory
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(self.device, use_cache=True)
        if self.collision_sampler is None:
            self.collision_sampler = FrankaCollisionSampler(
                self.device, with_base_link=False
            )
        rollout = self.rollout(batch, ROLLOUT_LENGTH, self.sample)
        
        train_loss, goal_loss, colli_loss, position_loss, rotation_loss = self.eval_rollout(
            rollout, batch
        )
        
        self.log("train_loss", train_loss)
        self.log("goal_loss", goal_loss)
        self.log("colli_loss", colli_loss)
        self.log("position_loss", position_loss)
        self.log("rotation_loss", rotation_loss)
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
            assert rollout.shape == (B, ROLLOUT_LENGTH+1, 7)
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
                assert sdf_values.shape == (B, ROLLOUT_LENGTH+1, num_spheres)
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