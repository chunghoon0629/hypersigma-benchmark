"""
HyperSIGMA - Hyperspectral Intelligence Comprehension Foundation Model

Core module providing unified access to HyperSIGMA encoders and utilities.
"""

from .models.spat_vit import SpatialVisionTransformer
from .models.spec_vit import SpectralVisionTransformer

__version__ = "1.0.0"
__all__ = [
    "SpatialVisionTransformer",
    "SpectralVisionTransformer",
]
