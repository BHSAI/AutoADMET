from pathlib import Path

import pandas as pd

PIPELINE_REPO_PATH = "bcrp_classify"


def main():
    datasets = pd.read_csv(
        Path(__file__).parent.parent / "data" / "dataset_config.csv"
    )["DATASET"]

    for dataset in datasets:
        # Load aggregate cross validation performance row for the top validating vNN config
        top_vnn_aggregate_stats = (
            pd.read_csv(f"output/vnn/leaderboards/leaderboard.{dataset}.morgan_fp.csv")
            .sort_values("kappa", ascending=False)
            .iloc[0]
        )
        # Load vNN per-fold cross validation performance for only the top validating config
        top_vnn_cross_val_stats = pd.read_csv(
            f"output/vnn/leaderboards/leaderboard-per_forld.{dataset}.morgan_fp.csv"
        )
        top_vnn_cross_val_stats = top_vnn_cross_val_stats[
            top_vnn_cross_val_stats["SmoothFactor"]
            == top_vnn_aggregate_stats["SmoothFactor"]
        ].reset_index(drop=True)

        # Configure the columns to match those in the validation step of the AutoML pipeline
        column_map = {
            "AreaUnderTheCurve": "ROC_AUC",
            "PR_AUC": "PR_AUC",
            "Accuracy": "Accuracy",
            "Sensitivity": "Sensitivity",
            "Specificity": "Specificity",
            "GMean": "GMean",
            "Precision": "Precision",
            "F1": "F1",
            "MCC": "MCC",
            "kappa": "Kappa",
        }
        top_vnn_cross_val_stats = top_vnn_cross_val_stats.rename(column_map, axis=1)
        top_vnn_cross_val_stats = top_vnn_cross_val_stats[
            [*column_map.values(), "Repeat", "Fold"]
        ]
        top_vnn_cross_val_stats["Descriptor"] = "Morgan"
        top_vnn_cross_val_stats["Model"] = "vNN"

        # Add the vNN per-fold CV stats to the AutoADMET per-fold CV stats
        autoadmet_cross_val_stats_file = (
            f"output/autoadmet_pipeline/{dataset}/cv_results/per_fold_results.csv"
        )
        autoadmet_cross_val_stats = pd.read_csv(autoadmet_cross_val_stats_file)
        if not (autoadmet_cross_val_stats["Model"] == "vNN").any():
            autoadmet_cross_val_stats = pd.concat(
                [autoadmet_cross_val_stats, top_vnn_cross_val_stats]
            )
            autoadmet_cross_val_stats.to_csv(
                autoadmet_cross_val_stats_file, index=False
            )

        # Add the vNN aggregate CV stats to the AutoADMET aggregate CV stats
        top_vnn_aggregate_stats = top_vnn_aggregate_stats.rename(column_map)
        top_vnn_aggregate_stats = top_vnn_aggregate_stats[list(column_map.values())]
        top_vnn_aggregate_stats["Descriptor"] = "Morgan"
        top_vnn_aggregate_stats["Model"] = "vNN"

        autoadmet_aggregate_stats_file = (
            f"output/autoadmet_pipeline/{dataset}/cv_results/results_summary.csv"
        )
        autoadmet_aggregate_stats = pd.read_csv(autoadmet_aggregate_stats_file)
        if not (autoadmet_aggregate_stats["Model"] == "vNN").any():
            autoadmet_aggregate_stats = pd.concat(
                [autoadmet_aggregate_stats, pd.DataFrame(top_vnn_aggregate_stats).T]
            )
            autoadmet_aggregate_stats.to_csv(
                autoadmet_aggregate_stats_file, index=False
            )

        # Rerun the AutoML cross validation performance visualization step. 
        import subprocess
        subprocess.run(
            f'{PIPELINE_REPO_PATH}/.venv/Scripts/python.exe {PIPELINE_REPO_PATH}/pipeline.py \
                --output "output/autoadmet_pipeline/{dataset}/cv_results" \
                --plot-only',
        ).check_returncode()


if __name__ == "__main__":
    main()
