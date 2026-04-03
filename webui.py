#!/usr/bin/env python3
"""Web UI demo: MNIST classification and Machine Translation."""

import argparse

import gradio as gr
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from mnist.mnist_transformer.config import TrainConfig as MNISTConfig
from mnist.mnist_transformer.data import MNIST_MEAN, MNIST_STD
from mnist.mnist_transformer.model import SequenceTransformerClassifier
from mnist.mnist_transformer.utils import get_device

from transformer.config import TrainConfig as TransformerConfig
from transformer.data import Vocabulary, get_dataloaders
from transformer.model import Transformer
from transformer.trainer import greedy_decode


# ============== MNIST ==============

def load_mnist_model(checkpoint_path: str, device: torch.device) -> SequenceTransformerClassifier:
    config = MNISTConfig()
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


def predict_mnist(image: dict, model: SequenceTransformerClassifier, device: torch.device) -> tuple[int, dict]:
    if image is None:
        return None, {}

    img = Image.fromarray(image["composite"]).convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)

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


# ============== Translation ==============

def load_transformer(checkpoint_path: str, device: torch.device):
    config = TransformerConfig()

    # Load vocab (quick load without full dataset)
    import os
    import pickle

    vocab_path = os.path.join(config.data_dir, "vocab.pkl")
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            src_vocab, tgt_vocab = pickle.load(f)
    else:
        # Build vocab from data
        _, _, src_vocab, tgt_vocab = get_dataloaders(
            data_dir=config.data_dir,
            batch_size=1,
            num_workers=0,
            max_len=config.max_len,
        )
        with open(vocab_path, "wb") as f:
            pickle.dump((src_vocab, tgt_vocab), f)

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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, src_vocab, tgt_vocab


def translate(text: str, model: Transformer, src_vocab: Vocabulary, tgt_vocab: Vocabulary, device: torch.device, max_len: int = 128) -> str:
    if not text.strip():
        return ""

    src = torch.tensor([src_vocab.encode(text)], dtype=torch.long).to(device)
    output = greedy_decode(
        model, src, max_len,
        tgt_vocab.sos_idx, tgt_vocab.eos_idx,
        src_vocab.pad_idx, device
    )
    return tgt_vocab.decode(output[0].tolist())


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Web UI Demo")
    parser.add_argument("--mnist-checkpoint", type=str, default="./mnist/checkpoints/best_model.pt")
    parser.add_argument("--transformer-checkpoint", type=str, default="./transformer/checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load models
    print("Loading MNIST model...")
    mnist_model = load_mnist_model(args.mnist_checkpoint, device)
    print("Loading Transformer model...")
    transformer_model, src_vocab, tgt_vocab = load_transformer(args.transformer_checkpoint, device)
    print("Models loaded!")

    # Build UI
    with gr.Blocks(title="Transformer Demo") as demo:
        gr.Markdown("# Transformer Demo")

        with gr.Tabs():
            # MNIST Tab
            with gr.TabItem("MNIST Digit Recognition"):
                gr.Markdown("Draw a digit (0-9) and click Predict")
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

                predict_btn.click(
                    fn=lambda img: predict_mnist(img, mnist_model, device),
                    inputs=canvas,
                    outputs=[pred_text, pred_label]
                )
                clear_btn.click(
                    fn=lambda: (None, None, {}),
                    outputs=[canvas, pred_text, pred_label]
                )

            # Translation Tab
            with gr.TabItem("German → English Translation"):
                gr.Markdown("Enter German text to translate to English")
                with gr.Row():
                    src_text = gr.Textbox(label="German (de)", placeholder="Ein Mann steht auf einer Straße...", lines=3)
                    tgt_text = gr.Textbox(label="English (en)", lines=3, interactive=False)
                translate_btn = gr.Button("Translate", variant="primary")

                translate_btn.click(
                    fn=lambda text: translate(text, transformer_model, src_vocab, tgt_vocab, device),
                    inputs=src_text,
                    outputs=tgt_text
                )

                gr.Examples(
                    examples=[
                        "ein mann steht auf einer straße",
                        "eine frau liest ein buch",
                        "kinder spielen im park",
                        "der hund läuft im garten",
                    ],
                    inputs=src_text,
                )

    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
