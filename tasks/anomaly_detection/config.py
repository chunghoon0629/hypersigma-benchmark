"""Configuration for anomaly detection task."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection experiments."""

    # Dataset settings
    dataset: str = 'pavia'  # 'pavia' or 'cri'

    # Model settings
    mode: str = 'ss'  # 'sa' (spatial-only) or 'ss' (spectral-spatial)

    # Normalization
    norm: str = 'std'  # 'std' or 'norm'
    norm_range: Tuple[int, int] = (-1, 1)

    # Input settings
    input_mode: str = 'part'  # 'whole' or 'part'
    input_size: List[int] = field(default_factory=lambda: [32, 32])
    overlap_size: int = 16

    # Training settings
    epochs: int = 10
    batch_size: int = 1
    val_batch_size: int = 1
    learning_rate: float = 6e-5
    weight_decay: float = 5e-4

    # Optimizer settings (layer decay)
    num_layers: int = 12
    layer_decay_rate: float = 0.9

    # Experiment settings
    num_runs: int = 1
    seed: int = 42
    print_freq: int = 3
    val_freq: int = 3

    # Labels
    ignore_label: int = 255

    # Workers
    num_workers: int = 2

    # Checkpointing
    use_checkpoint: bool = True


# Default configurations for different datasets
PAVIA_CONFIG = AnomalyConfig(
    dataset='pavia',
    input_size=[32, 32],
    overlap_size=16,
)

CRI_CONFIG = AnomalyConfig(
    dataset='cri',
    input_size=[64, 64],
    overlap_size=32,
)
