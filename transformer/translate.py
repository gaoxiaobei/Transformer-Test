#!/usr/bin/env python3
"""Translate text using trained Transformer model."""

import argparse

import torch

from transformer.config import TrainConfig
from transformer.data import get_dataloaders
from transformer.model import Transformer
from transformer.trainer import greedy_decode
from transformer.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Translate with Transformer")
    parser.add_argument("--checkpoint", type=str, default="./transformer/checkpoints/best_model.pt")
    parser.add_argument("--text", type=str, default=None, help="Text to translate (German)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    config = TrainConfig()
    device = get_device(args.device)

    # Load vocab
    _, _, src_vocab, tgt_vocab = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=1,
        num_workers=0,
        max_len=config.max_len,
    )

    # Load model
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

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {args.checkpoint}")
    print(f"Source: German (de), Target: English (en)")
    print("Enter text to translate (or --text '...')")
    print("-" * 50)

    def translate(text: str) -> str:
        src = torch.tensor([src_vocab.encode(text)], dtype=torch.long).to(device)
        output = greedy_decode(
            model, src, config.max_len,
            tgt_vocab.sos_idx, tgt_vocab.eos_idx,
            src_vocab.pad_idx, device
        )
        return tgt_vocab.decode(output[0].tolist())

    if args.text:
        print(f"Input:  {args.text}")
        print(f"Output: {translate(args.text)}")
    else:
        while True:
            try:
                text = input("de> ").strip()
                if not text:
                    continue
                result = translate(text)
                print(f"en> {result}")
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
