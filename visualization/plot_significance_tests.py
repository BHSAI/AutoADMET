from pathlib import Path

import pandas as pd

PIPELINE_REPO_PATH = "../bcrp_classify"


def main():
    datasets = pd.read_csv(Path(__file__).parent / "dataset_config.csv")["DATASET"]

    for dataset in datasets:
        top_vnn_aggregate_stats = (
            pd.read_csv(f"output/vnn/leaderboards/leaderboard.{dataset}.morgan_fp.csv")
            .sort_values("kappa", ascending=False)
            .iloc[0]
        )

        top_vnn_cross_val_stats = pd.read_csv(
            f"output/vnn/leaderboards/leaderboard.per_fold.{dataset}.morgan_fp.csv"
        )
        top_vnn_cross_val_stats = top_vnn_cross_val_stats[
            top_vnn_cross_val_stats["SmoothFactor"]
            == top_vnn_aggregate_stats["SmoothFactor"]
        ].reset_index(drop=True)
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
        top_vnn_cross_val_stats = top_vnn_cross_val_stats[list(column_map.values())]
        top_vnn_cross_val_stats["Descriptor"] = "Morgan"
        top_vnn_cross_val_stats["Model"] = "vNN"
        top_vnn_cross_val_stats["Repeat"] = 1
        top_vnn_cross_val_stats["Fold"] = top_vnn_cross_val_stats.index.to_numpy() + 1

        bhsai_cross_val_stats_file = (
            f"output/bhsai_automl_pipeline/{dataset}/cv_results/per_fold_results.csv"
        )
        bhsai_cross_val_stats = pd.read_csv(bhsai_cross_val_stats_file)
        if not (bhsai_cross_val_stats["Model"] == "vNN").any():
            bhsai_cross_val_stats = pd.concat(
                [bhsai_cross_val_stats, top_vnn_cross_val_stats]
            )
            bhsai_cross_val_stats.to_csv(bhsai_cross_val_stats_file, index=False)

        top_vnn_aggregate_stats = top_vnn_aggregate_stats.rename(column_map)
        top_vnn_aggregate_stats = top_vnn_aggregate_stats[list(column_map.values())]
        top_vnn_aggregate_stats["Descriptor"] = "Morgan"
        top_vnn_aggregate_stats["Model"] = "vNN"

        bhsai_aggregate_stats_file = (
            f"output/bhsai_automl_pipeline/{dataset}/cv_results/results_summary.csv"
        )
        bhsai_aggregate_stats = pd.read_csv(bhsai_aggregate_stats_file)
        if not (bhsai_aggregate_stats["Model"] == "vNN").any():
            bhsai_aggregate_stats = pd.concat(
                [bhsai_aggregate_stats, pd.DataFrame(top_vnn_aggregate_stats).T]
            )
            bhsai_aggregate_stats.to_csv(bhsai_aggregate_stats_file, index=False)

        import subprocess

        subprocess.run(
            f'../bcrp_classify/.venv/Scripts/python.exe ../bcrp_classify/pipeline.py \
                --output "output/bhsai_automl_pipeline/{dataset}/cv_results" \
                --plot-only',
        ).check_returncode()


if __name__ == "__main__":
    main()
