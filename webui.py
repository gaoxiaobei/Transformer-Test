#!/usr/bin/env python3
"""Web UI demo: draw a digit and get model prediction."""

import argparse

import gradio as gr
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from mnist_transformer.config import TrainConfig
from mnist_transformer.data import MNIST_MEAN, MNIST_STD
from mnist_transformer.model import SequenceTransformerClassifier
from mnist_transformer.utils import get_device


def load_model(checkpoint_path: str, device: torch.device) -> SequenceTransformerClassifier:
    config = TrainConfig()
    model = SequenceTransformerClassifier(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(image: dict, model: SequenceTransformerClassifier, device: torch.device) -> tuple[int, dict]:
    if image is None:
        return None, {}

    img = Image.fromarray(image["composite"]).convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)  # MNIST is white background, black digit

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred = torch.argmax(probs).item()
    label_probs = {str(i): float(probs[i]) for i in range(10)}
    return pred, label_probs


def main():
    parser = argparse.ArgumentParser(description="Web UI MNIST demo")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    print(f"Model loaded on {device}")

    def predict_fn(image):
        return predict(image, model, device)

    with gr.Blocks(title="MNIST Transformer Demo") as demo:
        gr.Markdown("# MNIST Transformer Demo\nDraw a digit (0-9) and click Predict")
        with gr.Row():
            canvas = gr.Sketchpad(
                label="Draw here",
                type="numpy",
                image_mode="L",
                brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", default_size=15),
                layers=False,
                width=280,
                height=280,
            )
            with gr.Column():
                pred_label = gr.Label(label="Prediction")
                pred_text = gr.Number(label="Predicted digit", precision=0)
        with gr.Row():
            clear_btn = gr.Button("Clear")
            predict_btn = gr.Button("Predict", variant="primary")

        predict_btn.click(fn=predict_fn, inputs=canvas, outputs=[pred_text, pred_label])
        clear_btn.click(fn=lambda: (None, None, {}), outputs=[canvas, pred_text, pred_label])

    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
