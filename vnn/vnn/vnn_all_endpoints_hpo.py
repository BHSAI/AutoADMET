from confidenceinterval import bootstrap
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np
import logging
import itertools as it
import sklearn.metrics as metrics
import time
import vnn.vnn_admet as vnn_admet
from vnn.variable_nearest_neighbor import VariableNearestNeighborsClassifier
from scipy.sparse import coo_array
import argparse

DATASETS = [
    "ames",
    "cyp1a2",
    "cyp2c9",
    "cyp2c19",
    "cyp2d6",
    "cyp3a4",
    "cytotox",
    "dili",
    "hlm",
    "mmp",
    "pgp_inhibitors",
]
FEATURIZATIONS = ["morgan_fp"]


def load_data(
    dataset: str,
    featurization: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    train_csv = f"data/preprocessed/{dataset}/{featurization}.train.csv"
    test_csv = f"data/preprocessed/{dataset}/{featurization}.test.csv"
    train = pd.read_csv(train_csv)
    train.columns = train.columns.astype(np.str_)
    test = pd.read_csv(test_csv)
    test.columns = test.columns.astype(np.str_)

    class_col = "CLASS"
    feature_cols = [col for col in train.columns if "FEATURE_" in col]
    X_train = train[feature_cols].copy()
    y_train = train[class_col].to_numpy()
    X_test = test[feature_cols].copy()
    y_test = test[class_col].to_numpy()

    return X_train, y_train, X_test, y_test


def conf_interval_dict(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    key: str,
    metric: Callable,
    **kwargs,
) -> dict:
    ci = bootstrap.bootstrap_ci(
        y_true=y_test.tolist(),
        y_pred=y_pred.tolist(),
        metric=lambda y_pred, y_true: metric(y_pred, y_true, **kwargs),
    )
    return {
        key: ci[0],
        f"{key}-lb": ci[1][0],
        f"{key}-ub": ci[1][1],
    }


def save_top_model_metrics(
    grid_search_results: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    top_performing_metrics_file: str,
):
    best_model_config = grid_search_results.iloc[0]
    best_model = VariableNearestNeighborsClassifier(best_model_config["SmoothFactor"])
    best_model.fit(X_train, y_train)

    start = time.perf_counter()
    y_pred: np.ndarray = best_model.predict(X_test)
    pred_time = time.perf_counter() - start

    performance = {
        "smoothing_factor": best_model_config["SmoothFactor"],
        **conf_interval_dict(y_test, y_pred, "kappa", metrics.cohen_kappa_score),
        **conf_interval_dict(y_test, y_pred, "accuracy", metrics.accuracy_score),
        **conf_interval_dict(
            y_test, y_pred, "recall", metrics.recall_score, pos_label=1
        ),
        **conf_interval_dict(
            y_test, y_pred, "specificity", metrics.recall_score, pos_label=0
        ),
        "kappa_val": grid_search_results.iloc[0]["kappa"],
        "pred_time": pred_time,
    }
    pd.DataFrame([performance]).to_csv(top_performing_metrics_file, index=False)


def main(use_prefit: bool):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting")

    for dataset, featurization in it.product(DATASETS, FEATURIZATIONS):
        logger.info(f"Working on {dataset} {featurization} data")

        # Load data
        logger.info(f"{dataset} {featurization} - Loading in preprocessed data")
        X_train, y_train, X_test, y_test = load_data(dataset, featurization)

        # Get filenames
        config_id_str = f"{dataset}.{featurization}"
        leaderboard_file = f"vnn/leaderboards/leaderboard.{config_id_str}.csv"
        plot_file = f"vnn/plots/plot.{config_id_str}.png"
        top_performing_metrics_file = f"vnn/top_models/top_model.{config_id_str}.csv"
        for directory in ["leaderboards", "plots", "top_models"]:
            Path(f"vnn/{directory}").mkdir(exist_ok=True)

        # Fit the automl object
        logger.info(f"{dataset} {featurization} - Hyperparameter optimizing vNN")
        if not use_prefit:
            grid_search_results = vnn_admet.run_grid_search(
                coo_array(X_train),
                y_train,
                smoothing_factor_space_resolution=0.005,
                tanimoto_threshold_space_resolution=1,
            )
        else:
            grid_search_results = pd.read_csv(leaderboard_file).sort_values(
                "SmoothFactor"
            )

        # Make training history plot
        logger.info(
            f"{dataset} {featurization} - Saving performance to time plot for model history"
        )
        ax = grid_search_results.plot(y="kappa", x="SmoothFactor")
        ax.figure.savefig(plot_file)  # type: ignore

        # Get performance metrics for the top model
        grid_search_results = grid_search_results.sort_values("kappa", ascending=False)
        logger.info(
            f"{dataset} {featurization} - Saving performance metrics for the top model"
        )
        grid_search_results.to_csv(leaderboard_file, index=False)
        save_top_model_metrics(
            grid_search_results,
            X_train,
            y_train,
            X_test,
            y_test,
            top_performing_metrics_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_prefit",
        action="store_true",
        help="Set this flag to skip training when regenerating performance data.",
    )
    args = parser.parse_args()

    main(args.use_prefit)
