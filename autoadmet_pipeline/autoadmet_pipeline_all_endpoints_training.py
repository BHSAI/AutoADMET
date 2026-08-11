from typing import Callable

import numpy as np
from sklearn import metrics
import logging
import pandas as pd
from pathlib import Path
from confidenceinterval import bootstrap
import argparse
import subprocess

PIPELINE_REPO_PATH = "./bcrp_classify"

datasets = pd.read_csv(Path(__file__).parent / "data" / "dataset_config.csv")[
    "DATASET"
].to_list()
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


def get_pred_times(final_model_dir: str) -> tuple[float, float, float, float]:
    pred_time_best, pred_time_best_normalized = (
        pd.read_csv(f"{final_model_dir}/best_model_test_metrics.csv", index_col=0)
        .T[["Prediction Time", "Prediction Time Normalized"]]
        .loc["Value"]
    )
    pred_time_ensemble, pred_time_ensemble_normalized = (
        pd.read_csv(f"{final_model_dir}/ensemble_test_metrics.csv", index_col=0)
        .T[["Prediction Time", "Prediction Time Normalized"]]
        .loc["Value"]
    )

    return (
        pred_time_best,
        pred_time_best_normalized,
        pred_time_ensemble,
        pred_time_ensemble_normalized,
    )  # type: ignore


def save_top_model_metrics(
    output_path: str,
    top_performing_metrics_file: str,
    time_bench_mark_tests: list[str],
):
    """
    Save the metrics we are interested in to a CSV file for both the top performing single model and the final ensemble.
    All test metrics are reported as confidence intervals.

    Args:
        cv_output_dir (str): The path to the output directory for the cross validation step.
        final_model_output_dir (str): The path to the output directory for the final model training step.
        top_performing_metrics_file (str): The path for the file to save the metrics to.
    """
    final_model_dir = f"{output_path}/final_model"
    cv_results_dir = f"{output_path}/cv_results"

    pred_df_best = pd.read_csv(f"{final_model_dir}/best_model_test_predictions.csv")
    pred_df_ensemble = pd.read_csv(f"{final_model_dir}/ensemble_test_predictions.csv")

    (
        pred_time_best,
        pred_time_best_normalized,
        pred_time_ensemble,
        pred_time_ensemble_normalized,
    ) = get_pred_times(final_model_dir)

    extra_time_benchmarks_best = {}
    extra_time_benchmarks_ensemble = {}
    for key in time_bench_mark_tests:
        (
            extra_time_benchmarks_best[f"pred_time_{key}"],
            extra_time_benchmarks_best[f"pred_time_{key}_normalized"],
            extra_time_benchmarks_ensemble[f"pred_time_{key}"],
            extra_time_benchmarks_ensemble[f"pred_time_{key}_normalized"],
        ) = get_pred_times(f"{final_model_dir}-{key}")

    # Get validation metrics
    kappa_val_ensemble = pd.read_csv(
        f"{final_model_dir}/ensemble_oof_metrics.csv", index_col=0
    ).loc["Kappa", "Value"]

    best_models_df = (
        pd.read_csv(f"{cv_results_dir}/results_summary.csv")
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

    with open(f"{cv_results_dir}/train_time.txt", "r") as file:
        train_time = float(file.read())

    with open(f"{final_model_dir}/ensemble_train_time.txt", "r") as file:
        ensemble_train_time = float(file.read())

    performance = [
        {
            "details": details_best,
            **test_performance_conf_intervals(
                pred_df_best["True_Label"].to_numpy(),
                pred_df_best["Predicted_Label"].to_numpy(),
            ),
            "kappa_val": kappa_val_best,
            "train_time": train_time,
            "pred_time": pred_time_best,
            "pred_time_normalized": pred_time_best_normalized,
            **extra_time_benchmarks_best,
        },
        {
            "details": details_ensemble,
            **test_performance_conf_intervals(
                pred_df_ensemble["True_Label"].to_numpy(),
                pred_df_ensemble["Predicted_Label"].to_numpy(),
            ),
            "kappa_val": kappa_val_ensemble,
            "train_time": train_time + ensemble_train_time,
            "pred_time": pred_time_ensemble,
            "pred_time_normalized": pred_time_ensemble_normalized,
            **extra_time_benchmarks_ensemble,
        },
    ]
    pd.DataFrame(performance).to_csv(top_performing_metrics_file, index=False)


def main(mode: str, pipeline_repo_path: str):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting")
    for dataset in datasets:
        logger.info(f"Working on {dataset} data")

        # Get filenames
        train_data_path = f"data/preprocessed/{dataset}/mordred_desc.train.csv"
        test_data_path = f"data/preprocessed/{dataset}/mordred_desc.test.csv"
        output_path = f"output/autoadmet_pipeline/{dataset}"
        top_performing_metrics_file = (
            f"top_models/autoadmet_pipeline/top_model.{dataset}.csv"
        )
        Path("output/autoadmet_pipeline").mkdir(exist_ok=True, parents=True)
        Path("top_models/autoadmet_pipeline").mkdir(exist_ok=True, parents=True)

        if mode != "parse-results-only":
            if mode == "full":
                logger.info(f"{dataset} - Fitting model")
                subprocess.run(
                    f'{pipeline_repo_path}/.venv/bin/python {pipeline_repo_path}/pipeline.py \
                        --input "{train_data_path}" \
                        --output "{output_path}/cv_results"',
                    shell=True,
                ).check_returncode()

            if mode == "load-model":
                logger.info(f"{dataset} - Loading and evaluating existing model")
                subprocess.run(
                    f'{pipeline_repo_path}/.venv/bin/python {pipeline_repo_path}/train_test_best_model.py \
                        --test "{test_data_path}" \
                        --load-model "{output_path}/final_model" \
                        --output "{output_path}/final_model"',
                    shell=True,
                ).check_returncode()
            else:
                logger.info(f"{dataset} - Refitting model with all data and evaluating")
                subprocess.run(
                    f'{pipeline_repo_path}/.venv/bin/python {pipeline_repo_path}/train_test_best_model.py \
                        --train "{train_data_path}" \
                        --test "{test_data_path}" \
                        --results "{output_path}/cv_results" \
                        --hyperparams "{output_path}/cv_results/optimized_hyperparameters.json" \
                        --output "{output_path}/final_model" \
                        --metric "Kappa" \
                        --ensemble',
                    shell=True,
                ).check_returncode()

            time_benchmark_datasets = [
                "small_compounds",
                "large_compounds",
                "representative-2500",
                "representative-5000",
                "representative-10000",
                "representative-20000",
            ]
            for time_benchmark_dataset in time_benchmark_datasets:
                logger.info(
                    f"{dataset} - Evaluating runtime on {time_benchmark_dataset} dataset"
                )
                subprocess.run(
                    f'{pipeline_repo_path}/.venv/bin/python {pipeline_repo_path}/train_test_best_model.py \
                        --test "data/preprocessed/time_benchmark/{time_benchmark_dataset}.mordred_desc.csv" \
                        --load-model "{output_path}/final_model" \
                         --output "{output_path}/final_model-{time_benchmark_dataset}"',
                    shell=True,
                ).check_returncode()

        # Get performance metrics for the top model
        logger.info(f"{dataset} - Saving performance metrics for the top model")
        save_top_model_metrics(
            output_path=output_path,
            top_performing_metrics_file=top_performing_metrics_file,
            time_bench_mark_tests=[
                "small_compounds",
                "large_compounds",
                "representative-2500",
                "representative-5000",
                "representative-10000",
                "representative-20000",
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "skip-cv", "load-model", "parse-results-only"],
        default="full",
        help="Set this flag to skip training when regenerating performance data.",
    )
    parser.add_argument(
        "--pipeline_repo_path",
        default=PIPELINE_REPO_PATH,
        help="The path to the AutoADMET pipeline repository.",
    )
    args = parser.parse_args()

    main(args.mode, args.pipeline_repo_path)
