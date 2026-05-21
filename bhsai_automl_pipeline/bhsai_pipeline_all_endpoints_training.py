from typing import Callable

import numpy as np
from sklearn import metrics
import logging
import pandas as pd
from pathlib import Path
from confidenceinterval import bootstrap
import argparse

PIPELINE_REPO_PATH = "../bcrp_classify"

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
CLASS_COL = "CLASS"


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
        A dict with keys: "{key}", "{key}-lb", and "{key}-ub" and values for the metric and the lower and upper bounds for the confidence interval for the metric, respectively.

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


def test_performance_conf_intervals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Get the the test metrics as confidence intervals.

    Args:
        y_test (np.ndarray): The true test values.
        y_pred (np.ndarray): The predicted test values.
    """
    return {
        **conf_interval_dict(y_test, y_pred, "kappa", metrics.cohen_kappa_score),
        **conf_interval_dict(y_test, y_pred, "accuracy", metrics.accuracy_score),
        **conf_interval_dict(
            y_test, y_pred, "recall", metrics.recall_score, pos_label=1
        ),
        **conf_interval_dict(
            y_test, y_pred, "specificity", metrics.recall_score, pos_label=0
        ),
    }


def save_top_model_metrics(
    cv_output_dir: str,
    final_model_output_dir: str,
    top_performing_metrics_file: str,
):
    """
    Save the metrics we are interested in to a CSV file for both the top performing single model and the final ensemble.
    All test metrics are reported as confidence intervals.

    Args:
        cv_output_dir (str): The path to the output directory for the cross validation step.
        final_model_output_dir (str): The path to the output directory for the final model training step.
        top_performing_metrics_file (str): The path for the file to save the metrics to.
    """
    pred_df_best = pd.read_csv(
        f"{final_model_output_dir}/best_model_test_predictions.csv"
    )
    pred_df_ensemble = pd.read_csv(
        f"{final_model_output_dir}/ensemble_test_predictions.csv"
    )

    pred_time_best = pd.read_csv(
        f"{final_model_output_dir}/best_model_test_metrics.csv", index_col=0
    ).loc["Prediction Time", "Value"]
    pred_time_ensemble = pd.read_csv(
        f"{final_model_output_dir}/ensemble_test_metrics.csv", index_col=0
    ).loc["Prediction Time", "Value"]

    kappa_val_ensemble = pd.read_csv(
        f"{final_model_output_dir}/ensemble_oof_metrics.csv", index_col=0
    ).loc["Kappa", "Value"]

    best_models_df = (
        pd.read_csv(f"{cv_output_dir}/results_summary.csv")
        .sort_values("Kappa", ascending=False)
        .reset_index(drop=True)
    )
    kappa_val_best = best_models_df["Kappa"].iloc[0]
    details_best = "-".join(best_models_df.iloc[0][["Descriptor", "Model"]])
    details_ensemble = "Ensemble: " + ", ".join(
        [
            "-".join(tup[1:])
            for tup in best_models_df.iloc[:5][["Descriptor", "Model"]].itertuples()
        ]
    )

    performance = [
        {
            "details": details_best,
            **test_performance_conf_intervals(
                pred_df_best["True_Label"].to_numpy(),
                pred_df_best["Predicted_Label"].to_numpy(),
            ),
            "kappa_val": kappa_val_best,
            "pred_time": pred_time_best,
        },
        {
            "details": details_ensemble,
            **test_performance_conf_intervals(
                pred_df_ensemble["True_Label"].to_numpy(),
                pred_df_ensemble["Predicted_Label"].to_numpy(),
            ),
            "kappa_val": kappa_val_ensemble,
            "pred_time": pred_time_ensemble,
        },
    ]
    pd.DataFrame(performance).to_csv(top_performing_metrics_file, index=False)


def main(use_prefit: bool, pipeline_repo_path: str):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting")
    for dataset in DATASETS:
        logger.info(f"Working on {dataset} data")

        # Get filenames
        train_data_path = f"data/preprocessed/{dataset}/mordred_desc.train.csv"
        test_data_path = f"data/preprocessed/{dataset}/mordred_desc.test.csv"
        output_path = f"bhsai_automl_pipeline/output/{dataset}"
        top_performing_metrics_file = (
            f"bhsai_automl_pipeline/top_models/top_model.{dataset}.csv"
        )
        for directory in [
            "output",
            "top_models",
        ]:
            Path(f"bhsai_automl_pipeline/{directory}").mkdir(exist_ok=True)

        if not use_prefit:
            logger.info(f"{dataset} - Fitting model")
            import subprocess

            subprocess.run(
                f'{pipeline_repo_path}/.venv/Scripts/python.exe {pipeline_repo_path}/pipeline.py \
                    --input "{train_data_path}" \
                    --output "{output_path}/cv_results"'
            ).check_returncode()

            logger.info(f"{dataset} - Saving leaderboards")
            subprocess.run(
                f'{pipeline_repo_path}/.venv/Scripts/python.exe {pipeline_repo_path}/train_test_best_model.py \
                    --train "{train_data_path}" \
                    --test "{test_data_path}" \
                    --results "{output_path}/cv_results" \
                    --hyperparams "{output_path}/cv_results/optimized_hyperparameters.json" \
                    --output "{output_path}/final_model" \
                    --metric "Kappa" \
                    --ensemble'
            ).check_returncode()

        # Get performance metrics for the top model
        logger.info(f"{dataset} - Saving performance metrics for the top model")
        save_top_model_metrics(
            cv_output_dir=f"{output_path}/cv_results",
            final_model_output_dir=f"{output_path}/final_model",
            top_performing_metrics_file=top_performing_metrics_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_prefit",
        action="store_true",
        help="Set this flag to skip training when regenerating performance data.",
    )
    parser.add_argument(
        "--pipeline_repo_path",
        default=PIPELINE_REPO_PATH,
        help="The path to the BHSAI AutoML pipeline repository.",
    )
    args = parser.parse_args()

    main(args.use_prefit, args.pipeline_repo_path)
