import time
from typing import Callable

from autogluon.tabular import TabularPredictor
import numpy as np
from sklearn import metrics
from specificity_metric import ag_specificity_scorer
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.common import space
from vnn_model import VNNModel
import itertools as it
import logging
import pandas as pd
from pathlib import Path
from confidenceinterval import bootstrap
import argparse

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
CLASS_COL = "CLASS"
TIME_LIMIT = None
EVAL_METRIC = "quadratic_kappa"
EXTRA_METRICS = [
    metric
    for metric in ["accuracy", "recall", ag_specificity_scorer, "quadratic_kappa"]
    if metric != EVAL_METRIC
]


def load_data(
    dataset: str,
    featurization: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load in the preprocessed compound data for the specified dataset and featurization.

    Args:
        dataset (str): The key for the compound dataset.
        featurization (str): The key for the featurization.

    Returns:
        train, test (tuple[pd.DataFrame, pd.DataFrame]): The training and testing data as dataframes.
    """
    train_csv = f"data/preprocessed/{dataset}/{featurization}.train.csv"
    test_csv = f"data/preprocessed/{dataset}/{featurization}.test.csv"
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    feature_cols = [col for col in train.columns if "FEATURE_" in col]
    train = train[[CLASS_COL, *feature_cols]]
    test = test[[CLASS_COL, *feature_cols]]

    return train, test


def get_fitted_predictor(
    train: pd.DataFrame,
    featurization: str,
    predictor_file: str,
    use_prefit: bool,
) -> TabularPredictor:
    """
    Train and save or load an AutoGluon predictor.

    Args:
        train (pd.DataFrame): The training data. Ignored if use_prefit is set to true.
        featurization (str): The featurization used. Used to decide what models to
            train. Ignored if use_prefit is set to true.
        predictor_file (str): The file to save the trained predictor to or load it from.
        use_prefit (bool): If false, do the training as normal. If true, do not train a
            new predictor, try loading it from predictor_file.
    """

    custom_hyperparameters = get_hyperparameter_config("default")
    extra_models = {
        "KNN": {
            "algorithm": "brute",
            "p": 2,
            "n_neighbors": space.Int(1, 20, default=10),
            "weights": space.Categorical("uniform", "distance"),
            "ag_args_fit": {"ignored_type_group_special": []},
            "ag_args": {
                "hyperparameter_tune_kwargs": {
                    "num_trials": 20,
                    "scheduler": "local",
                    "searcher": "auto",
                },
            },
        },
        "LR": {},
        # "EBM": {}, # No module named 'interpret'
        # "TABM": {}, # Not enough memory
    }
    custom_hyperparameters.update(extra_models)

    if featurization == "morgan_fp":
        custom_hyperparameters[VNNModel] = {  # type: ignore
            "smoothing_factor": space.Real(0.01, 1.0, default=0.3),
            "ag_args": {
                "hyperparameter_tune_kwargs": {
                    "num_trials": 100,
                    "scheduler": "local",
                    "searcher": "auto",
                },
            },
        }

    if not use_prefit:
        predictor = TabularPredictor(
            label=CLASS_COL,
            path=predictor_file,
            eval_metric=EVAL_METRIC,
            log_to_file=True,
        ).fit(
            train_data=train,
            time_limit=TIME_LIMIT * 60 if TIME_LIMIT else None,  # type: ignore
            presets="best_quality",
            hyperparameters=custom_hyperparameters,  # type: ignore
            dynamic_stacking=False,
            num_stack_levels=0,  # Limit to prevent overfitting
            num_bag_folds=5,
            num_bag_sets=5,
            excluded_model_types=["NN_TORCH"],
        )
    else:
        predictor = TabularPredictor.load(predictor_file)
    return predictor


def conf_interval_dict(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    key: str,
    metric: Callable,
    **kwargs,
) -> dict:
    """
    Get the value of the provided metric function as a 95% confidence interval.

    Args:
        y_test (np.ndarray): The true test values.
        y_pred (np.ndarray): The predicted test values.
        key (str): The str key for the metric for representation in the output dict.
        metric (Callable): The metric function.

    Returns:
        A dict with keys: "{key}", "{key}-lb", and "{key}-ub" and values for the metric
            and the lower and upper bounds for the confidence interval for the metric,
            respectively.

    """
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


def save_leaderboards(
    predictor: TabularPredictor,
    test: pd.DataFrame,
    leaderboard_file: str,
    top_ensemble_leaderboard_file: str,
):
    """
    Save leaderboards for the test performance of the trained models.

    Args:
        predictor (TabularPredictor): The trained AutoGluon predictor.
        test (pd.DataFrame): The test data to evaluate the models on.
        leaderboard_file (str): The file to save the full leaderboard including all
            trained models to.
        top_ensemble_leaderboard_file (str): The file to save a culled leaderboard only
            showing the top model and its ancestors (if the top model is an ensemble).
    """
    leaderboard = predictor.leaderboard(
        extra_metrics=EXTRA_METRICS,
        data=test,
        extra_info=True,
    )
    leaderboard = leaderboard.sort_values("score_val", ascending=False)
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.to_csv(leaderboard_file)

    # Add the top performing model to the set to explore
    to_explore: set[str] = {leaderboard["model"][0]}
    top_performing_ensemble_models: set[str] = set()
    while to_explore:
        model = to_explore.pop()
        model_idx = leaderboard["model"] == model
        # Get all features of this model that are the outputs of other models
        features = {
            feature
            for feature in leaderboard["features"][model_idx].iloc[0]
            if "FEATURE_" not in feature
        }
        to_explore.update(features.difference(top_performing_ensemble_models))
        top_performing_ensemble_models.add(model)

    top_performing_ensemble_leaderboard = leaderboard[
        leaderboard["model"].isin(top_performing_ensemble_models)
    ]
    top_performing_ensemble_leaderboard.to_csv(top_ensemble_leaderboard_file)


def save_top_model_metrics(
    predictor: TabularPredictor,
    test: pd.DataFrame,
    top_performing_metrics_file: str,
):
    """
    Save the metrics we are interested in to a CSV file for the top performing model.
    All test metrics are reported as confidence intervals.

    Args:
        predictor (TabularPredictor): The model to evaluate.
        test (pd.DataFrame): The test data to evaluate the predictor on.
        top_performing_metrics_file (str): The path for the file to save the metrics to.
    """
    y_test = test[CLASS_COL].to_numpy()
    start = time.perf_counter()
    y_pred: np.ndarray = predictor.predict(test)  # type: ignore
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
        "kappa_val": predictor.leaderboard()["score_val"][0],
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
        train, test = load_data(dataset, featurization)

        # Get filenames
        time_limit_str = (
            f"{TIME_LIMIT}min" if TIME_LIMIT is not None else "no_time_limit"
        )
        config_id_str = f"{dataset}.{featurization}.{EVAL_METRIC}.{time_limit_str}"
        leaderboard_file = f"autogluon/leaderboards/leaderboard.{config_id_str}.csv"
        top_ensemble_leaderboard_file = (
            f"autogluon/leaderboards/leaderboard.{config_id_str}.top_ensemble.csv"
        )
        predictor_file = f"autogluon/models/model.{config_id_str}"
        top_performing_metrics_file = (
            f"autogluon/top_models/top_model.{config_id_str}.csv"
        )
        for directory in ["leaderboards", "models", "top_models"]:
            Path(f"autogluon/{directory}").mkdir(exist_ok=True)

        logger.info(f"{dataset} {featurization} - Fitting AutoGluon predictor")
        predictor = get_fitted_predictor(
            train, featurization, predictor_file, use_prefit
        )

        logger.info(f"{dataset} {featurization} - Saving leaderboards")
        save_leaderboards(
            predictor,
            test,
            leaderboard_file,
            top_ensemble_leaderboard_file,
        )

        # Get performance metrics for the top model
        logger.info(
            f"{dataset} {featurization} - Saving performance metrics for the top model"
        )
        save_top_model_metrics(
            predictor,
            test,
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
