"""
Task-specific heads for HyperSIGMA.

Provides modular heads for different downstream tasks:
- Anomaly Detection (spatial-only and spectral-spatial)
- Classification (spatial-only and spectral-spatial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .spat_vit import SpatialVisionTransformer
from .spec_vit import SpectralVisionTransformer


class LayerNorm2d(nn.Module):
    """2D Layer Normalization."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class AnomalyDetectionHead(nn.Module):
    """
    Spatial-only anomaly detection head.

    Uses SpatViT backbone for feature extraction and outputs 2-class segmentation.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        spat_weights: Optional[str] = None,
    ):
        super().__init__()

        print('Using SpatSIGMA backbone for anomaly detection')

        self.encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        self.conv_features = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv_fc = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        features = self.encoder(x)
        img_feature = sum(features)
        img_feature = self.conv_features(img_feature)
        output = self.conv_fc(img_feature)
        return output.squeeze(1)


class SSAnomalyDetectionHead(nn.Module):
    """
    Spectral-Spatial anomaly detection head.

    Combines SpatViT and SpecViT for joint spectral-spatial feature extraction.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        spat_weights: Optional[str] = None,
        spec_weights: Optional[str] = None,
        num_tokens: int = 100,
    ):
        super().__init__()

        print('Using HyperSIGMA (Spat+Spec) backbone for anomaly detection')

        self.spat_encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        self.spec_encoder = SpectralVisionTransformer(
            NUM_TOKENS=num_tokens,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.1,
            out_indices=[11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.spat_encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        if spec_weights:
            self.spec_encoder.init_weights(spec_weights)
            print(f'Loaded SpecViT weights from {spec_weights}')

        self.conv_features = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv_fc = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, bias=False)
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_spec = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 128, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Spatial features
        img_features = self.spat_encoder(x)
        img_feature = sum(img_features)
        img_feature = self.conv_features(img_feature)

        # Spectral features
        spec_features = self.spec_encoder(x)
        spec_feature = sum(spec_features)
        spec_feature = self.pool(spec_feature).view(b, -1)
        spec_weights = self.fc_spec(spec_feature).view(b, -1, 1, 1)

        # Spectral-spatial fusion
        ss_feature = (1 + spec_weights) * img_feature
        output = self.conv_fc(ss_feature)

        return output.squeeze(1)


class ClassificationHead(nn.Module):
    """
    Spatial-only classification head.

    Uses SpatViT backbone for feature extraction and outputs class logits.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        patch_size: int = 2,
        spat_weights: Optional[str] = None,
        model_size: str = 'base',
    ):
        super().__init__()

        print(f'Using SpatSIGMA ({model_size}) backbone for classification')

        # Model configuration based on size
        if model_size == 'base':
            embed_dim, depth, num_heads, interval = 768, 12, 12, 3
            out_indices = [3, 5, 7, 11]
        elif model_size == 'large':
            embed_dim, depth, num_heads, interval = 1024, 24, 16, 6
            out_indices = [7, 11, 15, 23]
        elif model_size == 'huge':
            embed_dim, depth, num_heads, interval = 1280, 32, 16, 8
            out_indices = [10, 15, 20, 31]
        else:
            raise ValueError(f"Unknown model_size: {model_size}")

        self.embed_dim = embed_dim

        self.encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=patch_size,
            drop_path_rate=0.1,
            out_indices=out_indices,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=interval,
            n_points=8
        )

        if spat_weights:
            self.encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        # Note: SpatialVisionTransformer FPN already outputs 128 channels
        # No dimension reduction needed

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        features = self.encoder(x)

        # Features are already 128 channels from FPN
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]

        # Global average pooling
        f1 = F.adaptive_avg_pool2d(f1, 1)
        f2 = F.adaptive_avg_pool2d(f2, 1)
        f3 = F.adaptive_avg_pool2d(f3, 1)
        f4 = F.adaptive_avg_pool2d(f4, 1)

        # Flatten and concatenate
        features = torch.cat([
            f1.view(b, -1),
            f2.view(b, -1),
            f3.view(b, -1),
            f4.view(b, -1),
        ], dim=1)

        output = self.classifier(features)
        return output


class SSClassificationHead(nn.Module):
    """
    Spectral-Spatial classification head.

    Combines SpatViT and SpecViT for joint spectral-spatial feature extraction.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_classes: int,
        patch_size: int = 2,
        spat_weights: Optional[str] = None,
        spec_weights: Optional[str] = None,
        model_size: str = 'base',
        num_tokens: int = 100,
    ):
        super().__init__()

        print(f'Using HyperSIGMA ({model_size}) backbone for classification')

        # Model configuration based on size
        if model_size == 'base':
            embed_dim, depth, num_heads, interval = 768, 12, 12, 3
            out_indices = [3, 5, 7, 11]
        elif model_size == 'large':
            embed_dim, depth, num_heads, interval = 1024, 24, 16, 6
            out_indices = [7, 11, 15, 23]
        elif model_size == 'huge':
            embed_dim, depth, num_heads, interval = 1280, 32, 16, 8
            out_indices = [10, 15, 20, 31]
        else:
            raise ValueError(f"Unknown model_size: {model_size}")

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # Spatial encoder
        self.spat_encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=patch_size,
            drop_path_rate=0.1,
            out_indices=out_indices,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=interval,
            n_points=8
        )

        # Spectral encoder
        self.spec_encoder = SpectralVisionTransformer(
            NUM_TOKENS=num_tokens,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.1,
            out_indices=out_indices,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=interval,
            n_points=8
        )

        if spat_weights:
            self.spat_encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        if spec_weights:
            self.spec_encoder.init_weights(spec_weights)
            print(f'Loaded SpecViT weights from {spec_weights}')

        # Note: SpatialVisionTransformer FPN already outputs 128 channels
        # No dimension reduction needed

        # Spectral attention weights
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_spec1 = nn.Sequential(
            nn.Linear(num_tokens, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec2 = nn.Sequential(
            nn.Linear(num_tokens, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec3 = nn.Sequential(
            nn.Linear(num_tokens, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec4 = nn.Sequential(
            nn.Linear(num_tokens, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, bias=False),
            nn.Sigmoid(),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Spatial features
        img_f1, img_f2, img_f3, img_f4 = self.spat_encoder(x)

        # Spectral features
        spec_features = self.spec_encoder(x)
        spec_feature = spec_features[0]
        spec_feature = self.pool(spec_feature).view(b, -1)

        # Spectral attention weights
        spec_w1 = self.fc_spec1(spec_feature).view(b, -1, 1, 1)
        spec_w2 = self.fc_spec2(spec_feature).view(b, -1, 1, 1)
        spec_w3 = self.fc_spec3(spec_feature).view(b, -1, 1, 1)
        spec_w4 = self.fc_spec4(spec_feature).view(b, -1, 1, 1)

        # SpatialVisionTransformer FPN already outputs 128 channels
        # No dimension reduction needed

        # Spectral-spatial fusion (img_f1-4 are already 128 channels from FPN)
        ss_f1 = (1 + spec_w1) * img_f1
        ss_f2 = (1 + spec_w2) * img_f2
        ss_f3 = (1 + spec_w3) * img_f3
        ss_f4 = (1 + spec_w4) * img_f4

        # Global average pooling
        ss_f1 = F.adaptive_avg_pool2d(ss_f1, 1)
        ss_f2 = F.adaptive_avg_pool2d(ss_f2, 1)
        ss_f3 = F.adaptive_avg_pool2d(ss_f3, 1)
        ss_f4 = F.adaptive_avg_pool2d(ss_f4, 1)

        # Flatten and concatenate
        features = torch.cat([
            ss_f1.view(b, -1),
            ss_f2.view(b, -1),
            ss_f3.view(b, -1),
            ss_f4.view(b, -1),
        ], dim=1)

        output = self.classifier(features)
        return output


class DoubleConv(nn.Module):
    """Double convolution block for change detection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ChangeDetectionHead(nn.Module):
    """
    Spatial-only change detection head.

    Uses SpatViT backbone for bi-temporal feature extraction and outputs binary change map.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        patch_size: int = 5,
        seg_patches: int = 1,
        spat_weights: Optional[str] = None,
    ):
        super().__init__()

        print('Using SpatSIGMA backbone for change detection')

        self.patch_size = patch_size
        self.embed_dim = 768

        self.encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=seg_patches,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=False,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        num_dim = 64
        # Note: SpatViT FPN outputs 128 channels
        self.conv_features1 = nn.Conv2d(128, num_dim, kernel_size=1, bias=False)
        self.conv_features2 = nn.Conv2d(128, num_dim, kernel_size=1, bias=False)
        self.conv_features3 = nn.Conv2d(128, num_dim, kernel_size=1, bias=False)
        self.conv_features4 = nn.Conv2d(128, num_dim, kernel_size=1, bias=False)

        self.conv1 = DoubleConv(num_dim, 64, kernel_size=1)
        self.conv2 = DoubleConv(num_dim, 64, kernel_size=1)
        self.conv3 = DoubleConv(num_dim, 64, kernel_size=3)
        self.conv4 = DoubleConv(num_dim, 64, kernel_size=3)

        in_planes = num_dim * 4
        if patch_size == 5:
            self.classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, 2, kernel_size=1)
            )
        elif patch_size == 15:
            self.classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, 2, kernel_size=1)
            )
        else:
            # Default classifier for other patch sizes
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_planes, 2)
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Extract features from both time points
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)

        # Process features through conv layers
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        x1_feats = [op(feat1[i]) for i, op in enumerate(ops)]
        x2_feats = [op(feat2[i]) for i, op in enumerate(ops)]

        # Compute difference features
        f1 = torch.abs(x1_feats[3] - x2_feats[3])
        f1 = self.conv1(f1)

        f2 = torch.abs(x1_feats[2] - x2_feats[2])
        f2 = self.conv2(f2)

        f3 = torch.abs(x1_feats[1] - x2_feats[1])
        f3 = self.conv3(f3)

        f4 = torch.abs(x1_feats[0] - x2_feats[0])
        f4 = self.conv4(f4)

        # Concatenate and classify
        f = torch.cat([f1, f2, f3, f4], dim=1)
        output = self.classifier(f)
        return output.squeeze()


class SSChangeDetectionHead(nn.Module):
    """
    Spectral-Spatial change detection head.

    Combines SpatViT and SpecViT for joint spectral-spatial bi-temporal change detection.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        patch_size: int = 5,
        seg_patches: int = 1,
        spat_weights: Optional[str] = None,
        spec_weights: Optional[str] = None,
        num_tokens: int = 144,
    ):
        super().__init__()

        print('Using HyperSIGMA (Spat+Spec) backbone for change detection')

        self.patch_size = patch_size
        self.num_tokens = num_tokens

        self.spat_encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=seg_patches,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=False,
            interval=3,
            n_points=8
        )

        self.spec_encoder = SpectralVisionTransformer(
            NUM_TOKENS=num_tokens,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.1,
            out_indices=[11],  # Use only final layer output
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=False,
            use_abs_pos_emb=False,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.spat_encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        if spec_weights:
            self.spec_encoder.init_weights(spec_weights)
            print(f'Loaded SpecViT weights from {spec_weights}')

        # Note: SpatViT FPN outputs 128 channels, we project to num_tokens for fusion
        self.conv_features1 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features2 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features3 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features4 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)

        self.fc_spec1 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec2 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec3 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec4 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        num_dim = 64
        self.conv1 = DoubleConv(num_tokens, 64, kernel_size=1)
        self.conv2 = DoubleConv(num_tokens, 64, kernel_size=1)
        self.conv3 = DoubleConv(num_tokens, 64, kernel_size=3)
        self.conv4 = DoubleConv(num_tokens, 64, kernel_size=3)

        in_planes = num_dim * 4
        if patch_size == 5:
            self.classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, 2, kernel_size=1)
            )
        elif patch_size == 15:
            self.classifier = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // 2, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 2, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, in_planes // 4, kernel_size=3),
                nn.ReLU(),
                nn.Conv2d(in_planes // 4, 2, kernel_size=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_planes, 2)
            )

    def forward_fusion(self, x: torch.Tensor):
        """Extract and fuse spatial-spectral features."""
        b, _, h, w = x.shape

        # Spatial features
        img_features = self.spat_encoder(x)

        # Project to num_tokens dimension
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        img_fea = [op(img_features[i]) for i, op in enumerate(ops)]

        # Spectral features - use only the last layer output
        spec_features = self.spec_encoder(x)
        spec_feature = spec_features[0]  # Single output from out_indices=[11]
        spec_feature = self.pool(spec_feature).view(b, -1)

        # Spectral attention weights
        ops_spec = [self.fc_spec1, self.fc_spec2, self.fc_spec3, self.fc_spec4]
        spec_weights = [op(spec_feature).view(b, -1, 1, 1) for op in ops_spec]

        # Spectral-spatial fusion
        ss_features = [(1 + spec_weights[i]) * img_fea[i] for i in range(4)]

        return ss_features

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Extract fused features from both time points
        feat1 = self.forward_fusion(x1)
        feat2 = self.forward_fusion(x2)

        # Compute difference features
        f1 = torch.abs(feat1[3] - feat2[3])
        f1 = self.conv1(f1)

        f2 = torch.abs(feat1[2] - feat2[2])
        f2 = self.conv2(f2)

        f3 = torch.abs(feat1[1] - feat2[1])
        f3 = self.conv3(f3)

        f4 = torch.abs(feat1[0] - feat2[0])
        f4 = self.conv4(f4)

        # Concatenate and classify
        f = torch.cat([f1, f2, f3, f4], dim=1)
        output = self.classifier(f)
        return output.squeeze()


class DenoisingHead(nn.Module):
    """
    Spectral-Spatial denoising head.

    Combines SpatViT and SpecViT for hyperspectral image denoising/reconstruction.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        spat_weights: Optional[str] = None,
        spec_weights: Optional[str] = None,
        num_tokens: int = 100,
    ):
        super().__init__()

        print('Using HyperSIGMA (Spat+Spec) backbone for denoising')

        self.in_channels = in_channels
        self.num_tokens = num_tokens

        self.spat_encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=2,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        self.spec_encoder = SpectralVisionTransformer(
            NUM_TOKENS=num_tokens,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.1,
            out_indices=[11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.spat_encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        if spec_weights:
            self.spec_encoder.init_weights(spec_weights)
            print(f'Loaded SpecViT weights from {spec_weights}')

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Spectral attention
        self.fc_spec = nn.Sequential(
            nn.Linear(num_tokens, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, in_channels, bias=False),
            nn.Sigmoid(),
        )

        # Decoder - FPN outputs 128 channels, need to fuse and reconstruct
        # Sum of 4 FPN features gives 128 channels
        self.conv_reconstruct = nn.Conv2d(128 + in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_tail = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Spatial features - SpatViT with FPN returns list of 4 features, each 128 channels
        img_features = self.spat_encoder(x)
        # Sum all FPN features
        img_feature = sum(img_features)  # [B, 128, H, W]

        # Spectral features
        spec_features = self.spec_encoder(x)
        spec_feature = sum(spec_features)
        spec_feature = self.pool(spec_feature).view(b, -1)
        spec_weights = self.fc_spec(spec_feature).view(b, -1, 1, 1)

        # Spectral-spatial fusion
        # Upsample spatial features to input size
        img_feature = F.interpolate(img_feature, size=(h, w), mode='bilinear', align_corners=False)

        # Apply spectral attention to input
        x_weighted = (1 + spec_weights) * x

        # Concatenate and reconstruct
        combined = torch.cat([x_weighted, img_feature], dim=1)
        out = self.conv_reconstruct(combined)
        out = self.conv_tail(out)

        return out


class TargetDetectionHead(nn.Module):
    """
    Spatial-only target detection head.

    Uses SpatViT backbone and target spectrum for hyperspectral target detection.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        spat_weights: Optional[str] = None,
    ):
        super().__init__()

        print('Using SpatSIGMA backbone for target detection')

        self.encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        # FPN outputs 128 channels
        self.conv_features = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        # Target spectrum embedding
        self.conv_ts = nn.Sequential(
            nn.Linear(in_channels, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128, bias=False),
        )

    def forward(self, x: torch.Tensor, target_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input HSI [B, C, H, W]
            target_spectrum: Target spectrum [B, C]
        Returns:
            Detection map [B, H, W]
        """
        b, _, h, w = x.shape

        # Spatial features
        features = self.encoder(x)
        img_feature = sum(features)
        img_feature = self.conv_features(img_feature)  # [B, 128, H, W]

        # Target spectrum features
        ts_feature = self.conv_ts(target_spectrum).unsqueeze(1)  # [B, 1, 128]

        # Compute detection score via correlation
        output = (ts_feature @ img_feature.view(b, -1, h * w)).view(b, -1, h, w)

        return output.squeeze(1)


class SSTargetDetectionHead(nn.Module):
    """
    Spectral-Spatial target detection head.

    Combines SpatViT and SpecViT with target spectrum for joint spectral-spatial target detection.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        spat_weights: Optional[str] = None,
        spec_weights: Optional[str] = None,
        num_tokens: int = 100,
    ):
        super().__init__()

        print('Using HyperSIGMA (Spat+Spec) backbone for target detection')

        self.spat_encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        self.spec_encoder = SpectralVisionTransformer(
            NUM_TOKENS=num_tokens,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.1,
            out_indices=[11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=True,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.spat_encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        if spec_weights:
            self.spec_encoder.init_weights(spec_weights)
            print(f'Loaded SpecViT weights from {spec_weights}')

        # FPN outputs 128 channels
        self.conv_features = nn.Conv2d(128, 128, kernel_size=1, bias=False)

        # Target spectrum embedding
        self.conv_ts = nn.Sequential(
            nn.Linear(in_channels, 128, bias=False),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128, bias=False),
        )

        # Spectral attention
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_spec = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 128, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, target_spectrum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input HSI [B, C, H, W]
            target_spectrum: Target spectrum [B, C]
        Returns:
            Detection map [B, H, W]
        """
        b, _, h, w = x.shape

        # Spatial features
        img_features = self.spat_encoder(x)
        img_feature = sum(img_features)
        img_feature = self.conv_features(img_feature)  # [B, 128, H, W]

        # Spectral features for attention
        spec_features = self.spec_encoder(x)
        spec_feature = spec_features[0]
        spec_feature = self.pool(spec_feature).view(b, -1)
        spec_weights = self.fc_spec(spec_feature).view(b, -1, 1, 1)

        # Spectral-spatial fusion
        ss_feature = (1 + spec_weights) * img_feature

        # Target spectrum features
        ts_feature = self.conv_ts(target_spectrum).unsqueeze(1)  # [B, 1, 128]

        # Compute detection score via correlation
        output = (ts_feature @ ss_feature.view(b, -1, h * w)).view(b, -1, h, w)

        return output.squeeze(1)


class SumToOne(nn.Module):
    """Softmax normalization to ensure abundances sum to one."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x * self.scale, dim=1)


class UnmixingDecoder(nn.Module):
    """Linear decoder for spectral unmixing (endmember multiplication)."""

    def __init__(self, num_bands: int, num_endmembers: int, kernel_size: int = 1):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.kernel_size = kernel_size

        # Endmember matrix as convolution weights
        self.decoder = nn.Conv2d(
            num_endmembers, num_bands,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, abundances: torch.Tensor) -> torch.Tensor:
        """Reconstruct HSI from abundances."""
        return self.decoder(abundances)

    def get_endmembers(self) -> torch.Tensor:
        """Get endmember matrix."""
        weights = self.decoder.weight.data  # [num_bands, num_em, k, k]
        if self.kernel_size > 1:
            # Average spatial kernel
            return weights.mean(dim=(2, 3)).T  # [num_em, num_bands]
        return weights.squeeze(-1).squeeze(-1).T


class UnmixingHead(nn.Module):
    """
    Spatial-only spectral unmixing head.

    Uses SpatViT backbone for abundance estimation.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_endmembers: int,
        spat_weights: Optional[str] = None,
        num_tokens: int = 100,
        scale: float = 1.0,
        kernel_size: int = 1,
    ):
        super().__init__()

        print(f'Using SpatSIGMA backbone for unmixing ({num_endmembers} endmembers)')

        self.num_endmembers = num_endmembers
        self.num_tokens = num_tokens

        self.encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=False,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        # Feature projection - FPN outputs 128 channels
        self.conv_features1 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features2 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features3 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features4 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)

        # Feature processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )
        self.smooth = nn.Conv2d(num_tokens * 4, num_tokens, kernel_size=3, padding=1)

        # Abundance head
        self.abundance_head = nn.Sequential(
            nn.Conv2d(num_tokens, num_endmembers, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_endmembers),
            nn.Dropout(0.2),
        )

        self.sum_to_one = SumToOne(scale)
        self.decoder = UnmixingDecoder(in_channels, num_endmembers, kernel_size)

    def get_abundances(self, x: torch.Tensor) -> torch.Tensor:
        """Extract abundance maps from input."""
        H, W = x.shape[2], x.shape[3]

        # Encode
        features = self.encoder(x)

        # Project features
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        feats = [op(features[i]) for i, op in enumerate(ops)]

        # Process
        p4 = self.conv1(feats[3])
        p3 = self.conv2(feats[2])
        p2 = self.conv3(feats[1])
        p1 = self.conv4(feats[0])

        # Concatenate and smooth
        combined = torch.cat([p1, p2, p3, p4], dim=1)
        combined = F.interpolate(combined, size=(H, W), mode='bilinear', align_corners=True)
        combined = self.smooth(combined)

        # Get abundances
        abund = self.abundance_head(combined)
        abund = self.sum_to_one(abund)

        return abund

    def forward(self, x: torch.Tensor):
        """Forward pass returning abundances and reconstructed HSI."""
        abundances = self.get_abundances(x)
        reconstructed = self.decoder(abundances)
        return abundances, reconstructed

    def get_endmembers(self) -> torch.Tensor:
        """Get learned endmember matrix."""
        return self.decoder.get_endmembers()


class SSUnmixingHead(nn.Module):
    """
    Spectral-Spatial spectral unmixing head.

    Combines SpatViT and SpecViT for joint spectral-spatial abundance estimation.
    """

    def __init__(
        self,
        img_size: int,
        in_channels: int,
        num_endmembers: int,
        spat_weights: Optional[str] = None,
        spec_weights: Optional[str] = None,
        num_tokens: int = 100,
        scale: float = 1.0,
        kernel_size: int = 1,
    ):
        super().__init__()

        print(f'Using HyperSIGMA (Spat+Spec) backbone for unmixing ({num_endmembers} endmembers)')

        self.num_endmembers = num_endmembers
        self.num_tokens = num_tokens

        self.spat_encoder = SpatialVisionTransformer(
            img_size=img_size,
            in_chans=in_channels,
            patch_size=1,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=True,
            use_abs_pos_emb=False,
            interval=3,
            n_points=8
        )

        self.spec_encoder = SpectralVisionTransformer(
            NUM_TOKENS=num_tokens,
            img_size=img_size,
            in_chans=in_channels,
            drop_path_rate=0.1,
            out_indices=[11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=False,
            use_abs_pos_emb=False,
            interval=3,
            n_points=8
        )

        if spat_weights:
            self.spat_encoder.init_weights(spat_weights)
            print(f'Loaded SpatViT weights from {spat_weights}')

        if spec_weights:
            self.spec_encoder.init_weights(spec_weights)
            print(f'Loaded SpecViT weights from {spec_weights}')

        # Feature projection - FPN outputs 128 channels
        self.conv_features1 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features2 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features3 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)
        self.conv_features4 = nn.Conv2d(128, num_tokens, kernel_size=1, bias=False)

        # Spectral attention
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_spec1 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec2 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec3 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )
        self.fc_spec4 = nn.Sequential(
            nn.Linear(num_tokens, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_tokens, bias=False),
            nn.Sigmoid(),
        )

        # Feature processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(num_tokens, num_tokens, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens),
            nn.Dropout(0.2),
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(num_tokens * 4, num_tokens * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_tokens * 2),
            nn.Dropout(0.2),
            nn.Conv2d(num_tokens * 2, num_tokens, kernel_size=1),
        )

        # Abundance head
        self.abundance_head = nn.Sequential(
            nn.Conv2d(num_tokens, num_endmembers, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_endmembers),
            nn.Dropout(0.2),
        )

        self.sum_to_one = SumToOne(scale)
        self.decoder = UnmixingDecoder(in_channels, num_endmembers, kernel_size)

    def forward_fusion(self, x: torch.Tensor):
        """Extract and fuse spatial-spectral features."""
        b, _, h, w = x.shape

        # Spatial features
        img_features = self.spat_encoder(x)

        # Project features
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        img_fea = [op(img_features[i]) for i, op in enumerate(ops)]

        # Spectral features
        spec_features = self.spec_encoder(x)
        spec_feature = spec_features[0]
        spec_feature = self.pool(spec_feature).view(b, -1)

        # Spectral attention weights
        ops_spec = [self.fc_spec1, self.fc_spec2, self.fc_spec3, self.fc_spec4]
        spec_weights = [op(spec_feature).view(b, -1, 1, 1) for op in ops_spec]

        # Spectral-spatial fusion
        ss_features = [(1 + spec_weights[i]) * img_fea[i] for i in range(4)]

        return ss_features

    def get_abundances(self, x: torch.Tensor) -> torch.Tensor:
        """Extract abundance maps from input."""
        H, W = x.shape[2], x.shape[3]

        # Get fused features
        feats = self.forward_fusion(x)

        # Process
        p4 = self.conv1(feats[3])
        p3 = self.conv2(feats[2])
        p2 = self.conv3(feats[1])
        p1 = self.conv4(feats[0])

        # Concatenate and smooth
        combined = torch.cat([p1, p2, p3, p4], dim=1)
        combined = F.interpolate(combined, size=(H, W), mode='bilinear', align_corners=True)
        combined = self.smooth(combined)

        # Get abundances
        abund = self.abundance_head(combined)
        abund = self.sum_to_one(abund)

        return abund

    def forward(self, x: torch.Tensor):
        """Forward pass returning abundances and reconstructed HSI."""
        abundances = self.get_abundances(x)
        reconstructed = self.decoder(abundances)
        return abundances, reconstructed

    def get_endmembers(self) -> torch.Tensor:
        """Get learned endmember matrix."""
        return self.decoder.get_endmembers()
