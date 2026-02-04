"""HyperSIGMA model architectures."""

from .spat_vit import SpatialVisionTransformer
from .spec_vit import SpectralVisionTransformer
from .task_heads import (
    AnomalyDetectionHead,
    ClassificationHead,
    SSAnomalyDetectionHead,
    SSClassificationHead,
    ChangeDetectionHead,
    SSChangeDetectionHead,
    DenoisingHead,
    TargetDetectionHead,
    SSTargetDetectionHead,
    UnmixingHead,
    SSUnmixingHead,
)

__all__ = [
    "SpatialVisionTransformer",
    "SpectralVisionTransformer",
    "AnomalyDetectionHead",
    "ClassificationHead",
    "SSAnomalyDetectionHead",
    "SSClassificationHead",
    "ChangeDetectionHead",
    "SSChangeDetectionHead",
    "DenoisingHead",
    "TargetDetectionHead",
    "SSTargetDetectionHead",
    "UnmixingHead",
    "SSUnmixingHead",
]
