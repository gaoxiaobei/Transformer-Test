import argparse

import torch
from torch import nn

from mnist.mnist_transformer.config import TrainConfig
from mnist.mnist_transformer.data import get_dataloaders
from mnist.mnist_transformer.model import SequenceTransformerClassifier
from mnist.mnist_transformer.trainer import fit
from mnist.mnist_transformer.utils import get_device, set_seed, should_pin_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer on MNIST")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig()

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.device is not None:
        config.device = args.device

    set_seed(config.seed)
    device = get_device(config.device)
    pin_memory = should_pin_memory(device)
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=pin_memory,
    )

    model = SequenceTransformerClassifier(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=config.epochs,
        checkpoint_dir=config.checkpoint_dir,
        patience=config.early_stopping_patience,
        grad_clip_norm=config.grad_clip_norm,
    )


if __name__ == "__main__":
    main()
