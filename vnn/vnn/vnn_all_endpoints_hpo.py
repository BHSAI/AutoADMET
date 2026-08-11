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

datasets = pd.read_csv(Path(__file__).parent / "data" / "dataset_config.csv")["DATASET"]
FEATURIZATIONS = ["morgan_fp"]
CLASS_COL = "CLASS"


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

    feature_cols = [col for col in train.columns if "FEATURE_" in col]
    X_train = train[feature_cols].copy()
    y_train = train[CLASS_COL].to_numpy()
    X_test = test[feature_cols].copy()
    y_test = test[CLASS_COL].to_numpy()

    return X_train, y_train, X_test, y_test


def load_time_benchmark_data(featurization: str) -> dict[str, pd.DataFrame]:
    time_benchmark_data = {}
    for dataset in [
        "small_compounds",
        "large_compounds",
        "representative-2500",
        "representative-5000",
        "representative-10000",
        "representative-20000",
    ]:
        file = f"data/preprocessed/time_benchmark/{dataset}.{featurization}.csv"
        X_test = pd.read_csv(file)
        feature_cols = [col for col in X_test.columns if "FEATURE_" in col]
        X_test = X_test[feature_cols]
        time_benchmark_data[dataset] = X_test
    return time_benchmark_data


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


def predict_test_set(
    model: VariableNearestNeighborsClassifier,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    y_pred: np.ndarray = model.predict(X_test)
    pred_time = time.perf_counter() - start
    pred_time_normalized = pred_time / X_test.shape[0]

    return y_pred, pred_time, pred_time_normalized


def save_top_model_metrics(
    grid_search_results: pd.DataFrame,
    train_time: float,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    top_performing_metrics_file: str,
    time_benchmark_test_sets: dict[str, pd.DataFrame],
    pred_file: str,
):
    best_model_config = grid_search_results.iloc[0]
    best_model = VariableNearestNeighborsClassifier(
        best_model_config["SmoothFactor"], best_model_config["TanimotoDistance"]
    )
    best_model.fit(X_train, y_train)

    y_pred, test_pred_time, test_pred_time_norm = predict_test_set(best_model, X_test)
    np.save(pred_file, y_pred)

    extra_time_benchmarks = {
        f"pred_time_{key}_normalized": predict_test_set(best_model, test_set)[2]
        for key, test_set in time_benchmark_test_sets.items()
    }

    performance = {
        "smoothing_factor": best_model_config["SmoothFactor"],
        "distance_threshold": best_model_config["TanimotoDistance"],
        **conf_interval_dict(y_test, y_pred, "kappa", metrics.cohen_kappa_score),
        **conf_interval_dict(y_test, y_pred, "accuracy", metrics.accuracy_score),
        **conf_interval_dict(
            y_test, y_pred, "recall", metrics.recall_score, pos_label=1
        ),
        **conf_interval_dict(
            y_test, y_pred, "specificity", metrics.recall_score, pos_label=0
        ),
        "kappa_val": grid_search_results.iloc[0]["kappa"],
        "train_time": train_time,
        "pred_time": test_pred_time,
        "pred_time_normalized": test_pred_time_norm,
        **extra_time_benchmarks,
    }
    pd.DataFrame([performance]).to_csv(top_performing_metrics_file, index=False)


def main(use_prefit: bool):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting")

    for dataset, featurization in it.product(
        datasets["DATASET"], FEATURIZATIONS
    ):
        logger.info(f"Working on {dataset} {featurization} data")

        # Load data
        logger.info(f"{dataset} {featurization} - Loading in preprocessed data")
        X_train, y_train, X_test, y_test = load_data(dataset, featurization)

        # Get filenames
        config_id_str = f"{dataset}.{featurization}"
        leaderboard_file = f"output/vnn/leaderboards/leaderboard.{config_id_str}.csv"
        per_fold_leaderboard_file = (
            f"output/vnn/leaderboards/leaderboard-per_forld.{config_id_str}.csv"
        )
        train_time_file = f"output/vnn/leaderboards/train_time.{config_id_str}.txt"
        plot_file = f"output/vnn/plots/plot.{config_id_str}.png"
        top_performing_metrics_file = f"top_models/vnn/top_model.{config_id_str}.csv"
        pred_file = f"top_models/vnn/y_pred.{dataset}"
        for directory in ["leaderboards", "plots", "top_models"]:
            Path(f"output/vnn/{directory}").mkdir(exist_ok=True, parents=True)
        Path(f"top_models/vnn").mkdir(exist_ok=True, parents=True)

        # Fit the vnn model
        logger.info(f"{dataset} {featurization} - Hyperparameter optimizing vNN")
        if not use_prefit:
            smoothing_factor_search_space = vnn_admet.arange_inclusive(0.1, 1, 0.01)

            hpo_start = time.perf_counter()
            grid_search_results, per_fold_results = vnn_admet.vnn_search_provided_space(
                coo_array(X_train),
                y_train,
                smoothing_factor_space=smoothing_factor_search_space,
                tanimoto_threshold_space=[1],
                n_folds=5,
                n_repeats=5,
            )
            grid_search_results = grid_search_results.sort_values(
                "kappa", ascending=False
            )
            grid_search_results.to_csv(leaderboard_file, index=False)

            per_fold_results.to_csv(per_fold_leaderboard_file, index=False)
            train_time = time.perf_counter() - hpo_start
            with open(train_time_file, "w") as file:
                file.write(str(train_time))
        else:
            grid_search_results = pd.read_csv(leaderboard_file).sort_values(
                "kappa", ascending=False
            )
            with open(train_time_file, "r") as file:
                train_time = float(file.read())

        # Make training history plot
        logger.info(
            f"{dataset} {featurization} - Saving performance to smoothing factor plot"
        )
        ax = grid_search_results.sort_values("SmoothFactor").plot(
            y="kappa",
            x="SmoothFactor",
        )
        ax.axvline(
            grid_search_results.sort_values("kappa", ascending=False)["SmoothFactor"][
                0
            ],
            c="red",
        )
        ax.figure.savefig(plot_file)  # type: ignore

        # Get performance metrics for the top model
        logger.info(
            f"{dataset} {featurization} - Saving performance metrics for the top model"
        )
        time_benchmark_test_sets = load_time_benchmark_data(featurization)
        save_top_model_metrics(
            grid_search_results,
            train_time,
            X_train,
            y_train,
            X_test,
            y_test,
            top_performing_metrics_file,
            time_benchmark_test_sets,
            pred_file=pred_file,
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
