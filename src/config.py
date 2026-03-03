from dataclasses import dataclass

@dataclass
class TrainConfig:
    T: int = 1000
    batch_size: int = 128
    lr: float = 1e-4
    epochs: int = 30
    p_drop: float = 0.1
    num_classes: int = 10
    null_label: int = 10
    img_size: int = 28
    schedule: str = "linear"
    num_workers: int = 2
    ckpt_dir: str = "outputs/checkpoints"
    figures_dir: str = "outputs/figures"
    seed: int = 123
