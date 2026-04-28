from confidenceinterval import bootstrap
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np
from flaml import AutoML
from flaml.automl import data
import pickle
import vnn_estimator_flaml
from sklearn.preprocessing import MinMaxScaler
import logging
import itertools as it
import sklearn.metrics as metrics
import time
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

USE_PREFIT = True  # Set to true to regenerate plots and metrics without refitting
DATASETS = [
    "ames",
    "cytotox",
    "dili",
    "hlm",
    "mmp",
]
FEATURIZATIONS = [
    "morgan_fp",
    "mordred_desc",
]
TIME_LIMIT = 15  # in minutes
SEED = 7654321


def cohen_kappa(
    X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    config=None,
    groups_val=None,
    groups_train=None,
):
    y_pred = estimator.predict(X_val)
    return 1 - metrics.cohen_kappa_score(y_pred, y_val), {}


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

    if featurization == "mordred_desc":
        scaler = MinMaxScaler()
        scaler.fit(pd.concat([X_train, X_test]))

        X_train[X_train.columns] = scaler.transform(X_train)
        X_test[X_test.columns] = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def get_fitted_automl(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    featurization: str,
    log_file: str,
    pkl_file: str,
) -> AutoML:
    settings = {
        "time_budget": TIME_LIMIT * 60,  # total running time in seconds
        "task": "classification",  # task type
        "log_file_name": log_file,  # flaml log file
        "seed": SEED,  # random seed
        "ensemble": True,
        "metric": cohen_kappa,
        "eval_method": "cv",
        "log_type": "all",
    }

    automl = AutoML()

    if featurization == "morgan_fp":
        settings["estimator_list"] = [
            "lgbm",
            "rf",
            "xgboost",
            "extra_tree",
            "xgb_limitdepth",
            "sgd",
            "catboost",
            "lrl1",
            "vnn",
        ]
        automl.add_learner("vnn", vnn_estimator_flaml.VNNEstimator)

    if not USE_PREFIT:
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            **settings,
        )
        Path("flaml/models").mkdir(exist_ok=True)
        with open(pkl_file, "wb") as f:
            pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_file, "rb") as f:
            automl = pickle.load(f)

    return automl


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
        f"{key}-confidence_interval": f"({ci[1][0]:.2}, {ci[1][1]:.2})",
    }


def save_top_model_metrics(
    automl: AutoML,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    top_performing_metrics_file: str,
):
    start = time.perf_counter()
    y_pred: np.ndarray = automl.predict(X_test)  # type: ignore
    pred_time = time.perf_counter() - start

    performance = {
        **conf_interval_dict(y_test, y_pred, "kappa", metrics.cohen_kappa_score),
        **conf_interval_dict(y_test, y_pred, "accuracy", metrics.accuracy_score),
        **conf_interval_dict(
            y_test, y_pred, "recall", metrics.recall_score, pos_label=1
        ),
        **conf_interval_dict(
            y_test, y_pred, "specificity", metrics.recall_score, pos_label=0
        ),
        "kappa_val": 1 - automl.best_loss,
        "pred_time": pred_time,
    }
    pd.DataFrame([performance]).to_csv(top_performing_metrics_file)


def save_fit_history_plot(log_file: str, plot_file: str):
    (
        time_history,
        best_valid_loss_history,
        valid_loss_history,
        config_history,
        metric_history,
    ) = data.get_output_from_log(filename=log_file, time_budget=TIME_LIMIT * 60)
    estimator_list = [
        "lgbm",
        "rf",
        "xgboost",
        "extra",
        "xgb",
        "sgd",
        "catboost",
        "lrl1",
        "vnn",
    ]
    tableau = list(mcolors.TABLEAU_COLORS.values())[: len(estimator_list)]
    colors = {estimator: color for estimator, color in zip(estimator_list, tableau)}
    history = pd.DataFrame(
        [
            {
                "learner": config["Current Learner"],
                "time": time,
                "loss": loss,
            }
            for config, time, loss in zip(
                config_history, time_history, valid_loss_history
            )
        ]
    )
    plt.clf()
    plt.title("Learning Curve")
    plt.xlabel("Wall Clock Time (s)")
    plt.ylabel("Validation Kappa")
    for learner, color in colors.items():
        df_slice = history[history["learner"] == learner]
        if df_slice.shape[0]:
            plt.scatter(
                x=df_slice["time"],
                y=1 - df_slice["loss"],
                color=color,
                label=learner,
            )
    plt.step(
        time_history,
        1 - np.array(best_valid_loss_history),
        where="post",
        c=mcolors.CSS4_COLORS["black"],
    )
    plt.legend()
    plt.savefig(plot_file)


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting")

    for dataset, featurization in it.product(DATASETS, FEATURIZATIONS):
        logger.info(f"Working on {dataset} {featurization} data")

        # Load data
        logger.info(f"{dataset} {featurization} - Loading in preprocessed data")
        X_train, y_train, X_test, y_test = load_data(dataset, featurization)

        # Get filenames
        config_id_str = f"{dataset}.{featurization}.{TIME_LIMIT}min"
        log_file = f"flaml/logs/{config_id_str}.log"
        pkl_file = f"flaml/models/model.{config_id_str}.pkl"
        top_performing_metrics_file = f"flaml/top_models/top_model.{config_id_str}.csv"
        plot_file = f"flaml/plots/plot.{config_id_str}.png"
        for directory in ["logs", "models", "top_models", "plots"]:
            Path(f"flaml/{directory}").mkdir(exist_ok=True)

        # Fit the automl object
        logger.info(f"{dataset} {featurization} - Fitting the FLAML automl object")
        automl = get_fitted_automl(X_train, y_train, featurization, log_file, pkl_file)

        # Get performance metrics for the top model
        logger.info(
            f"{dataset} {featurization} - Saving performance metrics for the top model"
        )
        save_top_model_metrics(automl, X_test, y_test, top_performing_metrics_file)

        # Make training history plot
        logger.info(
            f"{dataset} {featurization} - Saving performance to time plot for model history"
        )
        save_fit_history_plot(log_file, plot_file)


if __name__ == "__main__":
    main()
