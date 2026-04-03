#!/usr/bin/env python3
"""Train the Transformer model."""

import argparse

import torch
import torch.nn as nn

from transformer.config import TrainConfig
from transformer.data import get_dataloaders
from transformer.model import Transformer
from transformer.trainer import LabelSmoothingLoss, WarmupLRScheduler, train_epoch, evaluate
from transformer.utils import set_seed, get_device, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer for Machine Translation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()

    # Override config with command line args
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.nhead is not None:
        config.nhead = args.nhead
    if args.num_layers is not None:
        config.num_encoder_layers = args.num_layers
        config.num_decoder_layers = args.num_layers

    set_seed(config.seed)
    device = get_device(config.device)
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, src_vocab, tgt_vocab = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_len=config.max_len,
    )

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    # Create model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        d_ff=config.d_ff,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = LabelSmoothingLoss(len(tgt_vocab), tgt_vocab.pad_idx, smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupLRScheduler(optimizer, config.d_model, config.warmup_steps)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        train_loss, _ = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            tgt_vocab.pad_idx, config.grad_clip_norm
        )
        val_loss, _ = evaluate(model, val_loader, criterion, device, tgt_vocab.pad_idx)

        print(f"Epoch {epoch + 1}/{config.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, f"{config.checkpoint_dir}/best_model.pt")
            print("  Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # Save last model
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, f"{config.checkpoint_dir}/last_model.pt")


if __name__ == "__main__":
    main()
