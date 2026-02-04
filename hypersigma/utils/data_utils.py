"""Data loading and preprocessing utilities."""

import os
from typing import Dict, Optional, Tuple, Any

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standard_normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range.

    Args:
        x: Input array.

    Returns:
        Normalized array.
    """
    max_value = np.max(x)
    min_value = np.min(x)
    if max_value == min_value:
        return x
    return (x - min_value) / (max_value - min_value)


def load_mat_data(
    filepath: str,
    data_key: str = None,
    gt_key: str = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load data from .mat file.

    Args:
        filepath: Path to the .mat file.
        data_key: Key for the data array (auto-detected if None).
        gt_key: Key for ground truth (optional).

    Returns:
        Tuple of (data, ground_truth) or (data, None).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    mat = sio.loadmat(filepath)

    # Auto-detect data key
    if data_key is None:
        # Common data keys
        for key in ['data', 'hsi', 'X', 'img', 'cube']:
            if key in mat:
                data_key = key
                break

    if data_key is None:
        # Use first non-private key
        for key in mat.keys():
            if not key.startswith('_'):
                data_key = key
                break

    data = mat[data_key].astype(np.float32)

    # Load ground truth if key specified
    gt = None
    if gt_key and gt_key in mat:
        gt = mat[gt_key].astype(np.float32)
    elif 'gt' in mat:
        gt = mat['gt'].astype(np.float32)
    elif 'groundT' in mat:
        gt = mat['groundT'].astype(np.float32)
    elif 'hsi_gt' in mat:
        gt = mat['hsi_gt'].astype(np.float32)

    return data, gt


def normalize_data(
    data: np.ndarray,
    method: str = 'std',
    feature_range: Tuple[float, float] = (-1, 1),
) -> np.ndarray:
    """
    Normalize hyperspectral data.

    Args:
        data: Input data of shape (H, W, C) or (N, C).
        method: Normalization method ('std' or 'minmax').
        feature_range: Range for MinMax normalization.

    Returns:
        Normalized data.
    """
    original_shape = data.shape

    # Reshape to 2D for sklearn
    if len(original_shape) == 3:
        h, w, c = original_shape
        data_2d = data.reshape(-1, c)
    else:
        data_2d = data

    if method == 'std':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    data_norm = scaler.fit_transform(data_2d)

    # Reshape back
    if len(original_shape) == 3:
        data_norm = data_norm.reshape(h, w, c)

    return data_norm


def create_patches(
    data: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    remove_zero_labels: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create patches from hyperspectral image.

    Args:
        data: Input data of shape (H, W, C).
        gt: Ground truth of shape (H, W).
        patch_size: Size of patches to extract.
        remove_zero_labels: Whether to exclude patches with label 0.

    Returns:
        Tuple of (patches, labels).
    """
    h, w, c = data.shape
    margin = patch_size // 2

    # Pad data
    padded_data = np.pad(
        data,
        ((margin, margin), (margin, margin), (0, 0)),
        mode='reflect'
    )

    patches = []
    labels = []

    for i in range(h):
        for j in range(w):
            label = gt[i, j]
            if remove_zero_labels and label == 0:
                continue

            patch = padded_data[i:i + patch_size, j:j + patch_size, :]
            patches.append(patch)
            labels.append(label)

    patches = np.array(patches)
    labels = np.array(labels)

    return patches, labels


def split_train_val_test(
    labels: np.ndarray,
    num_classes: int,
    train_num: int = 10,
    val_num: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train/val/test sets.

    Args:
        labels: Array of labels.
        num_classes: Number of classes.
        train_num: Number of training samples per class.
        val_num: Number of validation samples per class.
        seed: Random seed.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    np.random.seed(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for c in range(1, num_classes + 1):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        n_samples = len(class_indices)
        n_train = min(train_num, n_samples // 3)
        n_val = min(val_num, (n_samples - n_train) // 2)

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])

    return (
        np.array(train_indices),
        np.array(val_indices),
        np.array(test_indices),
    )


def apply_pca(data: np.ndarray, num_components: int) -> Tuple[np.ndarray, Any]:
    """
    Apply PCA to reduce spectral dimensions.

    Args:
        data: Input data of shape (H, W, C).
        num_components: Number of PCA components.

    Returns:
        Tuple of (reduced_data, pca_model).
    """
    from sklearn.decomposition import PCA

    h, w, c = data.shape
    data_2d = data.reshape(-1, c)

    pca = PCA(n_components=num_components)
    data_pca = pca.fit_transform(data_2d)
    data_pca = data_pca.reshape(h, w, num_components)

    return data_pca, pca
