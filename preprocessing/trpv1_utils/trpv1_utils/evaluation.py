"""
Model evaluation metrics and utilities.

This module provides functions for calculating various classification metrics:
- ROC-AUC, PR-AUC
- Accuracy, Precision, Recall (Sensitivity), Specificity
- F1-score, G-Mean
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Confusion matrix components (TP, TN, FP, FN)
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

# ============================================================================
# Core Metrics Calculation
# ============================================================================

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        y_prob: Predicted probabilities for positive class (array-like)

    Returns:
        dict: Dictionary of metrics with keys:
            - ROC_AUC: Area under ROC curve
            - PR_AUC: Area under Precision-Recall curve
            - Accuracy: Overall accuracy
            - Sensitivity: True positive rate (recall)
            - Specificity: True negative rate
            - GMean: Geometric mean of sensitivity and specificity
            - Precision: Positive predictive value
            - F1: F1-score
            - MCC: Matthews correlation coefficient
            - Kappa: Cohen's kappa
            - TP, TN, FP, FN: Confusion matrix components
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    # Sensitivity (Recall)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Geometric Mean
    gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0.0

    # ROC-AUC and PR-AUC (handle single-class probability vectors)
    if len(np.unique(y_prob)) == 1:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "GMean": gmean,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# ============================================================================
# Probability Extraction Utilities
# ============================================================================

def get_positive_class_probabilities(clf, X):
    """
    Extract predicted probabilities for the positive class (label=1).

    Handles edge cases where only one class appears in training data.

    Args:
        clf: Fitted classifier with predict_proba method
        X: Feature matrix

    Returns:
        numpy array of probabilities for positive class
    """
    proba = clf.predict_proba(X)
    classes = clf.classes_

    # Two classes present
    if len(classes) == 2:
        idx = np.where(classes == 1)[0]
        if len(idx) == 0:
            # Positive class unseen in training (rare edge case)
            return np.zeros(proba.shape[0], dtype=float)
        return proba[:, idx[0]]

    # Only one class learned
    elif len(classes) == 1:
        if classes[0] == 1:
            # Only positive class seen
            return np.ones(proba.shape[0], dtype=float)
        else:
            # Only negative class seen
            return np.zeros(proba.shape[0], dtype=float)

    # Empty classes (should not happen)
    else:
        return np.zeros(proba.shape[0], dtype=float)


# ============================================================================
# Metrics Aggregation
# ============================================================================

def aggregate_cv_metrics(fold_metrics):
    """
    Aggregate metrics across cross-validation folds.

    Args:
        fold_metrics: List of dictionaries, each containing metrics for one fold

    Returns:
        dict: Mean and std of each metric across folds
    """
    if not fold_metrics:
        return {}

    # Get all metric names
    metric_names = fold_metrics[0].keys()

    aggregated = {}
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]

        # Handle NaN values (e.g., from ROC-AUC with single-class predictions)
        values_array = np.array(values)
        valid_values = values_array[~np.isnan(values_array)]

        if len(valid_values) > 0:
            aggregated[f"{metric_name}_mean"] = np.mean(valid_values)
            aggregated[f"{metric_name}_std"] = np.std(valid_values)
        else:
            aggregated[f"{metric_name}_mean"] = np.nan
            aggregated[f"{metric_name}_std"] = np.nan

    return aggregated


def format_metrics_for_display(metrics, precision=4):
    """
    Format metrics dictionary for human-readable display.

    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places

    Returns:
        dict: Metrics with values rounded and formatted
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            formatted[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value):
                formatted[key] = "NaN"
            else:
                formatted[key] = round(value, precision)
        else:
            formatted[key] = value

    return formatted


# ============================================================================
# Performance Summary
# ============================================================================

def summarize_model_performance(y_true, y_pred, y_prob, model_name="Model"):
    """
    Generate comprehensive performance summary for a model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        model_name: Name of the model (for logging)

    Returns:
        dict: Comprehensive performance metrics
    """
    metrics = calculate_metrics(y_true, y_pred, y_prob)

    # Add summary statistics
    n_samples = len(y_true)
    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)

    summary = {
        "model_name": model_name,
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": n_positive / n_samples if n_samples > 0 else 0,
        **metrics
    }

    return summary
