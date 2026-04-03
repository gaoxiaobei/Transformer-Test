#!/usr/bin/env python3
"""Interactive demo: draw a digit and get model prediction."""

import argparse
import tkinter as tk
from tkinter import ttk

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms

from mnist_transformer.config import TrainConfig
from mnist_transformer.data import MNIST_MEAN, MNIST_STD
from mnist_transformer.model import SequenceTransformerClassifier
from mnist_transformer.utils import get_device


class DrawCanvas(tk.Canvas):
    """Canvas for drawing digits with mouse."""

    def __init__(self, master, width: int, height: int, brush_size: int = 15):
        super().__init__(master, width=width, height=height, bg="black", cursor="cross")
        self.brush_size = brush_size
        self.image = Image.new("L", (width, height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.bind("<B1-Motion>", self.paint)
        self.bind("<Button-1>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = self.brush_size
        self.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.delete("all")
        self.image = Image.new("L", (self.winfo_width(), self.winfo_height()), 0)
        self.draw = ImageDraw.Draw(self.image)

    def get_image(self) -> Image.Image:
        return self.image.resize((28, 28), Image.Resampling.LANCZOS)


class DemoApp:
    """Main application window."""

    def __init__(self, model: SequenceTransformerClassifier, device: torch.device):
        self.model = model
        self.device = device

        self.root = tk.Tk()
        self.root.title("MNIST Transformer Demo")
        self.root.resizable(False, False)

        # Canvas
        self.canvas = DrawCanvas(self.root, width=280, height=280)
        self.canvas.pack(padx=10, pady=10)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

        # Result
        self.result_var = tk.StringVar(value="Draw a digit (0-9)")
        ttk.Label(self.root, textvariable=self.result_var, font=("Arial", 16)).pack(pady=10)

        # Probability bars
        self.prob_frame = ttk.Frame(self.root)
        self.prob_frame.pack(pady=5, padx=20, fill=tk.X)
        self.prob_bars = []
        for i in range(10):
            frame = ttk.Frame(self.prob_frame)
            frame.pack(fill=tk.X, pady=1)
            ttk.Label(frame, text=str(i), width=2).pack(side=tk.LEFT)
            bar = ttk.Progressbar(frame, length=200, mode="determinate")
            bar.pack(side=tk.LEFT, padx=5)
            self.prob_bars.append(bar)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ])

    def predict(self):
        img = self.canvas.get_image()
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        pred = torch.argmax(probs).item()
        conf = probs[pred].item()
        self.result_var.set(f"Prediction: {pred} ({conf * 100:.1f}%)")

        for i, bar in enumerate(self.prob_bars):
            bar["value"] = probs[i].item() * 100

    def clear(self):
        self.canvas.clear()
        self.result_var.set("Draw a digit (0-9)")
        for bar in self.prob_bars:
            bar["value"] = 0

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Interactive MNIST demo")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded. Starting demo...")
    app = DemoApp(model, device)
    app.run()


if __name__ == "__main__":
    main()
