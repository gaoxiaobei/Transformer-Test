from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "./transformer/data"
    checkpoint_dir: str = "./transformer/checkpoints"
    seed: int = 42

    # Model (base configuration from paper)
    d_model: int = 512
    nhead: int = 8
    d_ff: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.1
    max_len: int = 128  # Multi30k sentences are short

    # Training
    batch_size: int = 128  # 4090 can handle larger batches
    num_workers: int = 8
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 4000
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5

    # Device
    device: str = "cuda"
