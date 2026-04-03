import argparse

import torch
from torchvision import datasets

from mnist_transformer.config import TrainConfig
from mnist_transformer.data import get_mnist_transforms
from mnist_transformer.model import SequenceTransformerClassifier
from mnist_transformer.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample inference on MNIST test set")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(device=args.device)
    device = get_device(config.device)

    model = SequenceTransformerClassifier(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=get_mnist_transforms(),
    )

    image, label = dataset[args.index]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    print(f"Index: {args.index}")
    print(f"True label: {label}")
    print(f"Pred label: {pred}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
