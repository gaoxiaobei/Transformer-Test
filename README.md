# Transformer Test

PyTorch implementations of Transformer architectures.

## Projects

### 1. MNIST Transformer Classification

A Transformer encoder that classifies MNIST handwritten digits by treating the image as a sequence of 784 tokens.

**Location:** [mnist/](mnist/)

**Features:**
- Serialize 28x28 image into 784 tokens
- Transformer Encoder classifier
- Train / evaluate / infer / interactive demo

**Quick Start:**
```bash
cd mnist
python train.py --epochs 10 --device cuda
```

### 2. Attention Is All You Need

A faithful implementation of the original Transformer model from the paper "Attention Is All You Need" (Vaswani et al., 2017).

**Location:** [transformer/](transformer/)

**Features:**
- Full encoder-decoder architecture
- Multi-head self-attention
- Positional encoding
- Label smoothing
- Learning rate warmup schedule
- Designed for 4090 GPU

**Architecture (Base model):**
- d_model: 512
- nhead: 8
- d_ff: 2048
- num_layers: 6 (encoder & decoder)

**Quick Start:**
```bash
cd transformer
python train.py --epochs 10 --batch-size 64
```

## Requirements

- Python >= 3.12
- PyTorch >= 2.9.1 with CUDA 12.4
- torchvision
- gradio (for web demo)

## Installation

```bash
# Install with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

## Project Structure

```
.
├── mnist/
│   ├── mnist_transformer/    # MNIST Transformer package
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── model.py
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── infer.py              # Single sample inference
│   └── demo.py               # Interactive GUI demo
│
├── transformer/
│   ├── config.py             # Training config
│   ├── data.py               # Data loading utilities
│   ├── model.py              # Transformer model
│   ├── trainer.py            # Training utilities
│   ├── utils.py              # Helper functions
│   └── train.py              # Training script
│
├── pyproject.toml
└── README.md
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Rush et al.
