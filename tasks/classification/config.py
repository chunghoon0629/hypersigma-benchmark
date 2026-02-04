"""Configuration for classification task."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ClassificationConfig:
    """Configuration for classification experiments."""

    # Dataset settings
    dataset: str = 'indian_pines'  # 'indian_pines', 'pavia_university', 'houston'

    # Model settings
    model_size: str = 'base'  # 'base', 'large', 'huge'
    patch_size: int = 2

    # Image settings
    img_size: int = 33
    pca_components: int = 30

    # Training settings
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 6e-5
    weight_decay: float = 0.05

    # Data split
    train_num: int = 10  # samples per class
    val_num: int = 5

    # Experiment settings
    num_runs: int = 10
    seed: int = 42

    # Optimizer settings
    num_layers: int = 12
    layer_decay_rate: float = 0.9


# Dataset-specific configurations
INDIAN_PINES_CONFIG = ClassificationConfig(
    dataset='indian_pines',
    pca_components=30,
    train_num=10,
)

PAVIA_U_CONFIG = ClassificationConfig(
    dataset='pavia_university',
    pca_components=30,
    train_num=10,
)

HOUSTON_CONFIG = ClassificationConfig(
    dataset='houston',
    pca_components=30,
    train_num=10,
)
