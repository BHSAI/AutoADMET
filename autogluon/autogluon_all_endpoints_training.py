from autogluon.tabular import TabularPredictor
from specificity_metric import ag_specificity_scorer
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.common import space
from vnn_model import VNNModel
import itertools as it
import logging
import pandas as pd
from pathlib import Path


USE_PREFIT = False

datasets = [
    "ames",
    "cytotox",
    "dili",
    "hlm",
    "mmp",
]

featurizations = [
    "morgan_fp",
    "mordred_desc",
]

time_limit = None
eval_metric = "quadratic_kappa"
extra_metrics = [
    metric
    for metric in ["accuracy", "recall", ag_specificity_scorer, "quadratic_kappa"]
    if metric != eval_metric
]


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting")
    for dataset, featurization in it.product(datasets, featurizations):
        logger.info(f"Working on {dataset} {featurization} data")

        logger.info(f"{dataset} {featurization} - Loading in preprocessed data")
        train_csv = f"data/preprocessed/{dataset}/{featurization}.train.csv"
        test_csv = f"data/preprocessed/{dataset}/{featurization}.test.csv"
        train = pd.read_csv(train_csv)
        test = pd.read_csv(test_csv)

        class_col = "CLASS"
        feature_cols = [col for col in train.columns if "FEATURE_" in col]
        train = train[[class_col, *feature_cols]]
        test = test[[class_col, *feature_cols]]

        time_limit_str = (
            f"{time_limit}min" if time_limit is not None else "no_time_limit"
        )
        config_id_str = f"{dataset}.{featurization}.{eval_metric}.{time_limit_str}"

        logger.info(f"{dataset} {featurization} - Configuring AutoGluon predictor")
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

        logger.info(f"{dataset} {featurization} - Fitting AutoGluon predictor")
        predictor_path = f"autogluon/models/model.{config_id_str}"
        if not USE_PREFIT:
            predictor = TabularPredictor(
                label=class_col,
                path=predictor_path,
                eval_metric=eval_metric,
                log_to_file=True,
            ).fit(
                train_data=train,
                time_limit=time_limit * 60 if time_limit else None,  # type: ignore
                presets="best_quality",
                hyperparameters=custom_hyperparameters,  # type: ignore
                dynamic_stacking=False,
                num_stack_levels=0,  # Limit to prevent overfitting
                num_bag_folds=5,
            )
        else:
            predictor = TabularPredictor.load(predictor_path)

        logger.info(f"{dataset} {featurization} - Creating leaderboard")
        leaderboard = predictor.leaderboard(
            extra_metrics=extra_metrics,
            data=test,
            extra_info=True,
        )
        leaderboard = leaderboard.sort_values("score_val", ascending=False)
        leaderboard = leaderboard.reset_index(drop=True)

        logger.info(f"{dataset} {featurization} - Saving leaderboard")
        Path("autogluon/leaderboards").mkdir(exist_ok=True)
        leaderboard.to_csv(f"autogluon/leaderboards/leaderboard.{config_id_str}.csv")

        logger.info(
            f"{dataset} {featurization} - Creating a leaderboard of models included in top performing ensemble"
        )
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

        logger.info(f"{dataset} {featurization} - Saving ensemble leaderboard")
        top_performing_ensemble_leaderboard.to_csv(
            f"autogluon/leaderboards/leaderboard.{config_id_str}.top_ensemble.csv"
        )


if __name__ == "__main__":
    main()
