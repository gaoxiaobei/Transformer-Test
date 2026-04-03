from dataclasses import dataclass


@dataclass
class TrainConfig:
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    batch_size: int = 64
    num_workers: int = 8
    val_split: float = 0.1

    d_model: int = 64
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    num_classes: int = 10

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5

    device: str = "cpu"
