# Transformer MNIST (PyTorch)

Use a sequence Transformer to classify MNIST handwritten digits.

## Features

- Serialize 28x28 image into 784 tokens
- Transformer Encoder based classifier
- Train / evaluate / infer scripts
- Best and last checkpoint saving
- CPU-first setup (GPU optional)

## Project Structure

- `mnist_transformer/config.py`: training config
- `mnist_transformer/data.py`: MNIST dataloaders
- `mnist_transformer/model.py`: sequence Transformer model
- `mnist_transformer/trainer.py`: train and validation loops
- `train.py`: training entry
- `evaluate.py`: test evaluation entry
- `infer.py`: single sample inference entry

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --epochs 10 --batch-size 64 --device cpu
```

Checkpoints are saved to `./checkpoints/best_model.pt` and `./checkpoints/last_model.pt`.

## Evaluate

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pt --device cpu
```

## Inference

```bash
python infer.py --checkpoint ./checkpoints/best_model.pt --index 0 --device cpu
```

## Expected Result

- After full training, test accuracy should typically reach around 97% to 98%.
- For quick smoke test, run `--epochs 1` to verify the pipeline.
