import argparse

import torch
from torch import nn

from mnist_transformer.config import TrainConfig
from mnist_transformer.data import get_dataloaders
from mnist_transformer.model import SequenceTransformerClassifier
from mnist_transformer.trainer import run_epoch
from mnist_transformer.utils import get_device, should_pin_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer on MNIST")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(batch_size=args.batch_size, device=args.device)
    device = get_device(config.device)
    pin_memory = should_pin_memory(device)

    _, _, test_loader = get_dataloaders(
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.4f}")


if __name__ == "__main__":
    main()
