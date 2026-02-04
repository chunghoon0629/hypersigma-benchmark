"""Evaluation metrics for HyperSIGMA benchmarks."""

import numpy as np
from sklearn import metrics
from typing import Dict, List, Tuple


def compute_auc_scores(output: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Compute AUC scores for anomaly detection.

    Args:
        output: Predicted anomaly scores.
        target: Ground truth labels.

    Returns:
        Dictionary containing multiple AUC metrics.
    """
    y_l = np.reshape(target, [-1, 1], order='F')
    y_p = np.reshape(output, [-1, 1], order='F')

    # Calculate ROC curve
    fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
    fpr = fpr[1:]
    tpr = tpr[1:]
    threshold = threshold[1:]

    # Compute various AUC metrics
    auc_roc = round(metrics.auc(fpr, tpr), 4)
    auc_fpr = round(metrics.auc(threshold, fpr), 4)
    auc_tpr = round(metrics.auc(threshold, tpr), 4)
    auc_combined = round(auc_roc + auc_tpr - auc_fpr, 4)
    auc_ratio = round(auc_tpr / auc_fpr, 4) if auc_fpr > 0 else float('inf')

    return {
        'auc_roc': auc_roc,
        'auc_fpr': auc_fpr,
        'auc_tpr': auc_tpr,
        'auc_combined': auc_combined,
        'auc_ratio': auc_ratio,
    }


def compute_classification_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    num_classes: int = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_pred: Predicted labels.
        y_true: Ground truth labels.
        num_classes: Number of classes (optional).

    Returns:
        Dictionary containing classification metrics.
    """
    # Overall accuracy
    overall_acc = metrics.accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    average_acc = np.mean(per_class_acc)

    # Kappa coefficient
    kappa = metrics.cohen_kappa_score(y_true, y_pred)

    # F1 score
    f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted')
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')

    return {
        'overall_accuracy': round(overall_acc, 4),
        'average_accuracy': round(average_acc, 4),
        'kappa': round(kappa, 4),
        'f1_weighted': round(f1_weighted, 4),
        'f1_macro': round(f1_macro, 4),
        'per_class_accuracy': per_class_acc.tolist(),
    }


def aa_and_each_accuracy(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate per-class accuracy and average accuracy from confusion matrix.

    Args:
        confusion_matrix: The confusion matrix.

    Returns:
        Tuple of (per-class accuracies, average accuracy).
    """
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)

    each_acc = np.nan_to_num(list_diag / list_raw_sum)
    average_acc = np.mean(each_acc)

    return each_acc, average_acc
