# mpinets/her_callback.py
import torch
import pytorch_lightning as pl
import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import gc


class HindsightExperienceReplayCallback(pl.Callback):
    def __init__(
        self,
        update_interval: int = 5,
        position_error_threshold: float = 0.05,
        rotation_error_threshold: float = 0.2,
        max_updates_per_epoch: int = 2000,
        max_samples_per_update: int = 2000,
        max_total_updates: int = 20000,
        enable_memory_optimizations: bool = True,
        skip_epoch_zero: bool = True,  # NEW: Skip epoch 0 by default
    ):
        super().__init__()
        self.update_interval = update_interval
        self.position_error_threshold = position_error_threshold
        self.rotation_error_threshold = rotation_error_threshold
        self.max_updates_per_epoch = max_updates_per_epoch
        self.max_samples_per_update = max_samples_per_update
        self.max_total_updates = max_total_updates
        self.enable_memory_optimizations = enable_memory_optimizations
        self.skip_epoch_zero = skip_epoch_zero  # NEW

        self.her_cache_dir = None
        self.total_updates = 0
        self.epoch_updates = []

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Setup HER cache at start of training"""
        if hasattr(trainer.datamodule, "data_train"):
            dataset = trainer.datamodule.data_train
            if hasattr(dataset, "_database"):
                original_path = Path(dataset._database)
                self.her_cache_dir = original_path.parent / "her_cache"
                self.her_cache_dir.mkdir(exist_ok=True)

                # Load existing HER updates
                her_cache_file = self.her_cache_dir / "her_updates.pkl"
                if her_cache_file.exists():
                    with open(her_cache_file, "rb") as f:
                        her_updates_numpy = pickle.load(f)

                    # Apply total updates limit
                    if len(her_updates_numpy) > self.max_total_updates:
                        print(
                            f"🎯 HER: Pruning cache from {len(her_updates_numpy)} to {self.max_total_updates} updates"
                        )
                        # Keep most recent updates
                        her_updates_numpy = dict(
                            list(her_updates_numpy.items())[-self.max_total_updates :]
                        )

                    # Convert numpy arrays to torch tensors
                    dataset.her_updates = {
                        k: torch.from_numpy(v).float()
                        for k, v in her_updates_numpy.items()
                    }
                    self.total_updates = len(dataset.her_updates)
                    print(
                        f"🎯 HER: Loaded {self.total_updates} existing updates from cache"
                    )
                else:
                    dataset.her_updates = {}
                    print("🎯 HER: Starting with no existing updates")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """After each training epoch, identify failed trajectories and update targets"""
        current_epoch = trainer.current_epoch

        # NEW: Skip epoch 0 if configured
        if self.skip_epoch_zero and current_epoch == 0:
            print(f"🎯 HER: Skipping epoch 0 (skip_epoch_zero={self.skip_epoch_zero})")
            return

        if current_epoch % self.update_interval != 0:
            return

        if not hasattr(trainer.datamodule, "data_train"):
            return

        print(f"🎯 HER: Running update at epoch {current_epoch}")
        self._update_failed_trajectories(pl_module, trainer.datamodule, current_epoch)

    def _memory_safe_rollout(self, pl_module, batch):
        """Perform rollout with true memory efficiency"""
        if self.enable_memory_optimizations:
            # Clear gradients and cache before rollout
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            pl_module.zero_grad()

        try:
            from mpinets.model_opt import ROLLOUT_LENGTH

            # CRITICAL: Use no_grad to prevent gradient computation
            with torch.no_grad():
                rollout = pl_module.rollout(
                    batch, ROLLOUT_LENGTH, pl_module.sample, unnormalize=True
                )
            return rollout
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("🎯 HER: OOM during rollout, skipping...")
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return None
            raise e

    def _update_failed_trajectories(self, pl_module, datamodule, current_epoch):
        """Identify failed trajectories and update their targets - TRUE MEMORY EFFICIENCY"""
        dataset = datamodule.data_train

        if not hasattr(dataset, "_database"):
            print("Warning: Dataset doesn't have direct database access for HER")
            return

        # Use true eval mode with no_grad
        pl_module.eval()

        # MEMORY FIX: Limit the number of samples we evaluate
        num_trajectories = min(len(dataset), self.max_samples_per_update)
        indices = np.random.choice(len(dataset), num_trajectories, replace=False)

        failed_indices = []
        new_targets = []
        position_errors = []
        rotation_errors = []

        # CRITICAL: Wrap the entire evaluation in no_grad
        with torch.no_grad():
            for i, idx in enumerate(indices):
                try:
                    # MEMORY FIX: Clear cache periodically
                    if self.enable_memory_optimizations and i % 50 == 0 and i > 0:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    batch = dataset[idx]

                    # Ensure batch is on correct device and has proper batch dimension
                    batch = {
                        k: v.unsqueeze(0) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                    batch = self._move_batch_to_device(batch, pl_module.device)

                    # Perform rollout with true memory safety
                    rollout = self._memory_safe_rollout(pl_module, batch)
                    if rollout is None:  # OOM occurred
                        continue

                    # Get final end-effector pose
                    final_config = rollout[-1]
                    final_pose = pl_module.fk_sampler.end_effector_pose(final_config)

                    # Compute errors
                    target_position = batch["target_position"]
                    target_rotation = batch["target_rotation"]

                    position_error = torch.linalg.vector_norm(
                        final_pose[:, :3, 3] - target_position, dim=1
                    ).item()

                    # Compute rotation error
                    pred_rotmat = final_pose[:, :3, :3]
                    target_rotmat = target_rotation.reshape(-1, 3, 3)

                    R_delta = torch.bmm(pred_rotmat, target_rotmat.transpose(1, 2))
                    trace = torch.diagonal(R_delta, dim1=-2, dim2=-1).sum(-1)
                    rotation_error = (
                        torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6))
                        .mean()
                        .item()
                    )

                    # Check if trajectory failed
                    position_failed = position_error > self.position_error_threshold
                    rotation_failed = rotation_error > self.rotation_error_threshold

                    if position_failed or rotation_failed:
                        failed_indices.append(idx)
                        position_errors.append(position_error)
                        rotation_errors.append(rotation_error)

                        # Use final achieved pose as new target
                        new_target_pose = torch.cat(
                            [
                                final_pose[0, :3, 3],  # position
                                final_pose[0, :3, :3].flatten(),  # rotation matrix
                            ]
                        )
                        new_targets.append(new_target_pose.cpu())

                    # MEMORY FIX: Early stopping
                    if len(failed_indices) >= self.max_updates_per_epoch:
                        break

                except Exception as e:
                    print(f"Error processing index {idx}: {e}")
                    continue

        # Clear VRAM after evaluation
        if self.enable_memory_optimizations:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Switch back to train mode
        pl_module.train()

        # Update dataset with new targets
        if failed_indices:
            self._update_dataset_targets(dataset, failed_indices, new_targets)

            # Calculate statistics
            avg_position_error = np.mean(position_errors) if position_errors else 0
            avg_rotation_error = np.mean(rotation_errors) if rotation_errors else 0
            position_failures = sum(
                1 for e in position_errors if e > self.position_error_threshold
            )
            rotation_failures = sum(
                1 for e in rotation_errors if e > self.rotation_error_threshold
            )

            print(f"🎯 HER Epoch {current_epoch}:")
            print(f"   • Evaluated {len(indices)} trajectories")
            print(f"   • Updated {len(failed_indices)} failed trajectories")
            print(
                f"   • Position failures: {position_failures}/{len(failed_indices)} "
                f"(avg error: {avg_position_error:.4f})"
            )
            print(
                f"   • Rotation failures: {rotation_failures}/{len(failed_indices)} "
                f"(avg error: {avg_rotation_error:.4f})"
            )
            print(f"   • Total HER updates: {self.total_updates}")

            # Track for summary
            self.epoch_updates.append(
                {
                    "epoch": current_epoch,
                    "evaluated": len(indices),
                    "updates": len(failed_indices),
                    "position_failures": position_failures,
                    "rotation_failures": rotation_failures,
                    "avg_position_error": avg_position_error,
                    "avg_rotation_error": avg_rotation_error,
                }
            )
        else:
            print(
                f"🎯 HER Epoch {current_epoch}: No failed trajectories found in {len(indices)} samples"
            )

    def _update_dataset_targets(self, dataset, indices, new_targets):
        """Persistently store HER updates with memory limits"""
        if self.her_cache_dir is None:
            return

        her_cache_file = self.her_cache_dir / "her_updates.pkl"

        # Load existing
        if her_cache_file.exists():
            with open(her_cache_file, "rb") as f:
                all_updates_numpy = pickle.load(f)

            # MEMORY FIX: Enforce total updates limit
            if len(all_updates_numpy) >= self.max_total_updates:
                # Remove oldest updates to make room
                excess = len(all_updates_numpy) - self.max_total_updates + len(indices)
                if excess > 0:
                    print(
                        f"🎯 HER: Removing {excess} oldest updates to stay under limit"
                    )
                    # Convert to list to maintain order and remove oldest
                    items = list(all_updates_numpy.items())
                    all_updates_numpy = dict(items[excess:])

            # Convert existing numpy arrays to torch tensors
            all_updates = {
                k: torch.from_numpy(v).float() for k, v in all_updates_numpy.items()
            }
        else:
            all_updates = {}

        # Count new unique updates
        new_unique_updates = 0
        for idx, new_target in zip(indices, new_targets):
            if idx not in all_updates:
                new_unique_updates += 1
            all_updates[int(idx)] = new_target

        # Update total count
        self.total_updates = len(all_updates)

        # Save as numpy for compatibility, but store as tensors in memory
        all_updates_numpy = {k: v.numpy() for k, v in all_updates.items()}
        with open(her_cache_file, "wb") as f:
            pickle.dump(all_updates_numpy, f)

        # Update in-memory cache with torch tensors
        dataset.her_updates = all_updates

    def _move_batch_to_device(self, batch, device):
        """Utility to move batch to device"""
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Print summary at the end of training"""
        if self.epoch_updates:
            print("\n🎯 HER Training Summary:")
            print("=" * 50)
            total_evaluated = sum(epoch["evaluated"] for epoch in self.epoch_updates)
            total_epoch_updates = sum(epoch["updates"] for epoch in self.epoch_updates)
            print(f"Total trajectories evaluated: {total_evaluated}")
            print(f"Total trajectories updated during training: {total_epoch_updates}")
            print(f"Final total HER updates in cache: {self.total_updates}")
            print(f"Cache size limit: {self.max_total_updates}")
            print(
                f"HER update intervals: {[epoch['epoch'] for epoch in self.epoch_updates]}"
            )
