"""
Training Utilities for Distributed Training
============================================
Shared utilities for tokenizer and predictor finetuning.
"""
import os
import random
import datetime
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


def setup_ddp() -> tuple[int, int, int]:
    """
    Initialize distributed data parallel environment.

    Returns
    -------
    tuple: (rank, world_size, local_rank)
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    print(f"[DDP Setup] Global Rank: {rank}/{world_size}, Local Rank (GPU): {local_rank} on device {torch.cuda.current_device()}")
    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int = 0):
    """Set random seed for reproducibility across all libraries."""
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    """Calculate trainable parameters and return human-readable string."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


def format_time(seconds: float) -> str:
    """Format duration in seconds to H:M:S string."""
    return str(datetime.timedelta(seconds=int(seconds)))


def create_dataloaders(
    config,
    train_dataset,
    val_dataset,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """Create distributed dataloaders for training and validation."""
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def save_checkpoint(model, path: str, rank: int = 0):
    """Save model checkpoint (only on rank 0)."""
    if rank == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save_pretrained(path)
        print(f"Checkpoint saved to {path}")


class CometLogger:
    """Simple Comet ML logger wrapper."""

    def __init__(self, config, rank: int = 0):
        self.rank = rank
        self.logger = None
        if rank == 0 and config.use_comet and config.comet_api_key:
            try:
                import comet_ml
                self.logger = comet_ml.Experiment(
                    api_key=config.comet_api_key,
                    project_name=config.comet_project_name,
                    workspace=config.comet_workspace,
                )
                self.logger.add_tag(config.comet_tag)
                self.logger.set_name(config.comet_name)
                self.logger.log_parameters(config.__dict__)
                print("Comet Logger Initialized.")
            except Exception as e:
                print(f"Comet init failed: {e}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None, epoch: Optional[int] = None):
        if self.logger and self.rank == 0:
            self.logger.log_metric(name, value, step=step, epoch=epoch)

    def log_metrics(self, metrics: dict, step: Optional[int] = None, epoch: Optional[int] = None):
        if self.logger and self.rank == 0:
            for name, value in metrics.items():
                self.logger.log_metric(name, value, step=step, epoch=epoch)

    def end(self):
        if self.logger and self.rank == 0:
            self.logger.end()


from typing import Optional