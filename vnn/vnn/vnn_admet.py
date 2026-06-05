from typing import Final, Literal

import numpy as np
from scipy.sparse import coo_array
from scipy import stats
import vnn.variable_nearest_neighbor as vnn_classifier
import pandas as pd
import itertools as it
import sklearn.metrics
from tqdm import tqdm

N_FOLDS = 10

CLASSIFICATION: Final = "classification"
REGRESSION: Final = "regression"
PROBLEM_TYPE = Literal["classification", "regression"]

NAME_COL: Final = "name"
SMILES_COL: Final = "smiles"
PROPERTY_COL: Final = "property"


def eval_classification_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    total_original = len(y_true)

    out_of_domain = np.isnan(y_pred)
    y_true = y_true[~out_of_domain]
    y_pred = y_pred[~out_of_domain]
    num_out_of_domain = out_of_domain.sum()
    coverage = 1 - num_out_of_domain / total_original

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel().tolist()

    total_pos = tp + fn
    total_neg = tn + fp
    total_within_domain = total_pos + total_neg

    sensitivity = tp / total_pos if total_pos else np.nan
    specificity = tn / total_neg if total_neg else np.nan
    accuracy = (tp + tn) / total_within_domain if total_within_domain else np.nan
    if total_within_domain:
        e = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (
            np.pow(total_within_domain, 2)
        )
        kappa = (accuracy - e) / (1 - e)
    else:
        kappa = np.nan
    tpr = tp / total_pos if total_pos else np.nan
    fpr = fp / total_neg if total_neg else np.nan
    roc = 0.5 * (1 - fpr + tpr) if total_pos and total_neg else np.nan

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "kappa": kappa,
        "Coverage": coverage,
        "AreaUnderTheCurve": roc,
    }


def eval_regression_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    total_original = len(y_true)

    out_of_domain = np.isnan(y_pred)
    y_true = y_true[~out_of_domain]
    y_pred = y_pred[~out_of_domain]
    num_out_of_domain = out_of_domain.sum()
    coverage = 1 - num_out_of_domain / total_original

    correlation = stats.pearsonr(y_true, y_pred).correlation

    return {
        "R": correlation,
        "Coverage": coverage,
    }


eval_metrics = {
    CLASSIFICATION: eval_classification_dict,
    REGRESSION: eval_regression_dict,
}


def v_neighbors_weighted_avg_oof(
    dist_matrix: np.ndarray,
    y: np.ndarray,
    distance_threshold: float,
    smoothing_factor: float,
    n_folds: int,
) -> np.ndarray:
    """
    Get the out-of-fold weighted averages of the v neighbors that pass the distance
    threshold for each query vector. Before predicting probabilities, each distance
    (i, j) where query vector i and training vector j are in the same fold
    (i.e., i %% n_folds == j %% n_folds) will be masked to NaN such that training
    vector j will be excluded from the weighted_average for query vector i.

    Args:
        dist_matrix (np.ndarray): An m x n matrix representing the
            distances between m query vectors and n training vectors.
        y (np.ndarray): An n x 1 matrix representing the truth values for the n
            training vectors.
        distance_threshold (float): The distance threshold.
        smoothing_factor (float): The smoothing factor.
        n_folds (int): The number of cross validation folds. Folds are assigned by
            the index of the vector mod n_folds.
    """
    # Get an n_folds x n_folds identity matrix and tile it to the size of the distance matrix.
    # Every cell (i, j) where i % n_folds == j % n_folds will be True.
    shape = np.array(dist_matrix.shape)
    same_fold = np.tile(
        np.eye(n_folds).astype(bool),
        tuple(np.ceil(shape / n_folds).astype(int)),
    )[: shape[0], : shape[1]]

    oof_tanimoto_dist = np.ma.masked_array(dist_matrix, same_fold).filled(np.nan)

    return vnn_classifier.v_neighbors_weighted_avg(
        dist_matrix=oof_tanimoto_dist,
        y=y,
        distance_threshold=distance_threshold,
        smoothing_factor=smoothing_factor,
    )


def run_grid_search(
    X: coo_array,
    y: np.ndarray,
    problem_type: PROBLEM_TYPE = CLASSIFICATION,
    smoothing_factor_space_resolution: float = 0.1,
    tanimoto_threshold_space_resolution: float = 0.1,
    n_folds=N_FOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform (smoothing_factor_space_resolution x tanimoto_threshold_space_resolution) grid search on
    smoothing factor and tanimoto distance threshold and report the evaluation metrics.

    Args:
        X (pd.DataFrame): An (n x k) dataframe where each row is a training vector and each column a feature.
        y (np.ndarray): An (n x 1) vector of truth values for the training vectors.
        problem_type (Literal["classification", "regression"]): The problem type.
        smoothing_factor_space_resolution (int): The resolution of the search space for smoothing factor.
        tanimoto_threshold_space_resolution (int): The resolution of the search space for distance threshold.

    Returns:
        A dataframe of the evaluated combinations of hyperparameters and their validation performance metrics.
    """
    tanimoto_dist = vnn_classifier.tanimoto_distance_matrix(X, X)

    # Perform grid search over smoothing factor and distance threshold
    if smoothing_factor_space_resolution <= 0.9:
        smoothing_factor_space = it.chain(
            np.arange(start=0.1, stop=1, step=smoothing_factor_space_resolution),
            [1],
        )
        smoothing_factor_space_resolution = 0.9 / smoothing_factor_space_resolution + 1
    else:
        smoothing_factor_space = [1]
        smoothing_factor_space_resolution = 1

    if tanimoto_threshold_space_resolution <= 0.9:
        tanimoto_threshold_space = it.chain(
            np.arange(start=0.1, stop=1, step=tanimoto_threshold_space_resolution),
            [1],
        )
        tanimoto_threshold_space_resolution = (
            0.9 / tanimoto_threshold_space_resolution + 1
        )
    else:
        tanimoto_threshold_space = [1]
        tanimoto_threshold_space_resolution = 1

    search_grid = it.product(smoothing_factor_space, tanimoto_threshold_space)
    results = []
    per_fold_eval_metrics = []
    for smoothing_factor, tanimoto_threshold in tqdm(
        search_grid,
        "Testing hyperparameters",
        int(smoothing_factor_space_resolution * tanimoto_threshold_space_resolution),
    ):
        y_pred = v_neighbors_weighted_avg_oof(
            dist_matrix=tanimoto_dist,
            y=y,
            distance_threshold=tanimoto_threshold,
            smoothing_factor=smoothing_factor,
            n_folds=n_folds,
        )
        if problem_type == CLASSIFICATION:
            y_pred = y_pred.round()

        results.append(
            {
                "SmoothFactor": smoothing_factor,
                "TanimotoDistance": tanimoto_threshold,
                **eval_metrics[problem_type](y, y_pred),
            }
        )

        per_fold_eval_metrics.extend(
            calc_per_fold_eval_metrics(
                y=y,
                y_pred=y_pred,
                n_folds=n_folds,
                smoothing_factor=smoothing_factor,
                tanimoto_threshold=tanimoto_threshold,
                problem_type=problem_type,
            )
        )

    return pd.DataFrame(results), pd.DataFrame(per_fold_eval_metrics)


def calc_per_fold_eval_metrics(
    y: np.ndarray,
    y_pred: np.ndarray,
    n_folds: int,
    smoothing_factor: float,
    tanimoto_threshold: float,
    problem_type: PROBLEM_TYPE = CLASSIFICATION,
) -> list[dict]:
    per_fold_eval_metrics: list[dict] = []
    for fold in range(n_folds):
        y_pred_fold = y_pred[fold::n_folds]
        y_fold = y[fold::n_folds]
        per_fold_eval_metrics.append(
            {
                "SmoothFactor": smoothing_factor,
                "TanimotoDistance": tanimoto_threshold,
                **eval_metrics[problem_type](y_fold, y_pred_fold),
            }
        )
    return per_fold_eval_metrics
