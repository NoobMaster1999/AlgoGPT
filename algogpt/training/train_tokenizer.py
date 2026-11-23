"""
Tokenizer Finetuning for Indian Derivatives
============================================
Fine-tunes the Kronos Tokenizer on NIFTY/BANKNIFTY data.
"""
import os
import sys
import json
import time
from time import gmtime, strftime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algogpt.training.config import FinetuneConfig, load_config_from_env
from algogpt.training.utils import (
    setup_ddp,
    cleanup_ddp,
    set_seed,
    get_model_size,
    format_time,
    create_dataloaders,
    save_checkpoint,
    CometLogger,
)
from algogpt.data.dataset import create_train_val_datasets
from algogpt.model import KronosTokenizer


def train_tokenizer(config: FinetuneConfig):
    """Main tokenizer finetuning loop."""
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config.seed, rank)

    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        symbols=config.symbols,
        data_dir=config.data_dir,
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        max_context=config.max_context,
        clip=config.clip,
        train_ratio=config.train_ratio,
        data_source=config.data_source,
        fetcher_kwargs=config.fetcher_kwargs,
        start_date=config.start_date,
        end_date=config.end_date,
        interval=config.interval,
        seed=config.seed,
    )

    train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        config, train_dataset, val_dataset, rank, world_size,
        batch_size=config.tokenizer_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Logger (rank 0 only)
    comet_logger = CometLogger(config, rank)
    master_summary = {}

    if rank == 0:
        os.makedirs(os.path.join(config.tokenizer_save_path, "checkpoints"), exist_ok=True)
        master_summary = {
            "start_time": strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
            "save_directory": config.tokenizer_save_path,
            "world_size": world_size,
            "config": config.__dict__,
        }

    dist.barrier()

    # Model initialization
    tokenizer = KronosTokenizer(
        d_in=config.tok_d_in,
        d_model=config.tok_d_model,
        n_heads=config.tok_n_heads,
        ff_dim=config.tok_ff_dim,
        n_enc_layers=config.tok_n_enc_layers,
        n_dec_layers=config.tok_n_dec_layers,
        ffn_dropout_p=config.tok_ffn_dropout_p,
        attn_dropout_p=config.tok_attn_dropout_p,
        resid_dropout_p=config.tok_resid_dropout_p,
        s1_bits=config.tok_s1_bits,
        s2_bits=config.tok_s2_bits,
        beta=config.tok_beta,
        gamma0=config.tok_gamma0,
        gamma=config.tok_gamma,
        zeta=config.tok_zeta,
        group_size=config.tok_group_size,
    )

    # Load pretrained weights
    print(f"[Rank {rank}] Loading pretrained tokenizer from {config.pretrained_tokenizer_path}")
    pretrained_state = torch.load(config.pretrained_tokenizer_path, map_location="cpu") if os.path.exists(config.pretrained_tokenizer_path) else None

    if pretrained_state:
        # Handle HF model format
        if "model_state_dict" in pretrained_state:
            pretrained_state = pretrained_state["model_state_dict"]
        tokenizer.load_state_dict(pretrained_state, strict=False)
        print(f"[Rank {rank}] Loaded pretrained weights")
    else:
        # Try loading from HF Hub
        try:
            tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
            print(f"[Rank {rank}] Loaded from HF Hub")
        except Exception as e:
            print(f"[Rank {rank}] Warning: Could not load pretrained tokenizer: {e}")

    tokenizer.to(device)
    tokenizer.train()

    tokenizer_ddp = DDP(tokenizer, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"Tokenizer Model Size: {get_model_size(tokenizer)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        tokenizer_ddp.parameters(),
        lr=config.tokenizer_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.tokenizer_epochs,
        pct_start=config.lr_pct_start,
        div_factor=config.lr_div_factor,
    )

    best_val_loss = float("inf")
    batch_idx_global = 0
    start_time = time.time()

    for epoch_idx in range(config.tokenizer_epochs):
        epoch_start_time = time.time()
        tokenizer_ddp.train()
        train_sampler.set_epoch(epoch_idx)
        train_dataset.set_epoch_seed(epoch_idx * 10000 + rank)
        val_dataset.set_epoch_seed(0)

        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Forward pass
            (z_pre, z), bsq_loss, quantized, z_indices = tokenizer_ddp(batch_x)
            loss = bsq_loss  # Tokenizer loss is just the BSQ loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer_ddp.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Logging
            if rank == 0 and (batch_idx_global + 1) % config.tokenizer_log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[Rank {rank}, Epoch {epoch_idx+1}/{config.tokenizer_epochs}, Step {i+1}/{len(train_loader)}] LR {lr:.6f}, Loss: {loss.item():.4f}")
                comet_logger.log_metric("train_tokenizer_loss", loss.item(), step=batch_idx_global)
                comet_logger.log_metric("train_bsq_loss", bsq_loss.item(), step=batch_idx_global)
                comet_logger.log_metric("tokenizer_lr", lr, step=batch_idx_global)

            batch_idx_global += 1

        # Validation
        tokenizer_ddp.eval()
        tot_val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                (z_pre, z), bsq_loss, quantized, z_indices = tokenizer_ddp(batch_x)
                tot_val_loss += bsq_loss.item()
                val_batches += 1

        # Reduce validation loss across ranks
        val_loss_tensor = torch.tensor(tot_val_loss, device=device)
        val_batches_tensor = torch.tensor(val_batches, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches_tensor, op=dist.ReduceOp.SUM)

        avg_val_loss = val_loss_tensor.item() / val_batches_tensor.item() if val_batches_tensor.item() > 0 else 0

        # Checkpointing (rank 0 only)
        if rank == 0:
            print(f"\n--- Epoch {epoch_idx+1}/{config.tokenizer_epochs} Summary ---")
            print(f"Validation Loss: {avg_val_loss:.6f}")
            print(f"Time This Epoch: {format_time(time.time() - epoch_start_time)}")
            print(f"Total Time Elapsed: {format_time(time.time() - start_time)}\n")
            comet_logger.log_metric("val_tokenizer_loss", avg_val_loss, epoch=epoch_idx)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(config.tokenizer_save_path, "checkpoints", "best_model")
                save_checkpoint(tokenizer_ddp.module, save_path, rank)
                print(f"Best tokenizer saved (Val Loss: {best_val_loss:.6f})")

        dist.barrier()

    if rank == 0:
        master_summary["final_result"] = {"best_val_loss": best_val_loss}
        with open(os.path.join(config.tokenizer_save_path, "summary.json"), "w") as f:
            json.dump(master_summary, f, indent=4, default=str)
        print("Tokenizer training finished. Summary saved.")
        comet_logger.end()

    cleanup_ddp()


if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with `torchrun`.")

    config = load_config_from_env()
    train_tokenizer(config)