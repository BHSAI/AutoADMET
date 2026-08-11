from pathlib import Path

from matplotlib.patches import Patch
import pandas as pd
import itertools as it
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from pathlib import Path
from collections import defaultdict
import adjustText

datasets = pd.read_csv(Path(__file__).parent.parent / "data" / "dataset_config.csv")
datasets = datasets.sort_values("FULL_SIZE").reset_index(drop=True)

SPREAD = 0.33
TICK_WIDTH = 0.08
METRICS = ["Kappa", "Accuracy", "Specificity", "GMean", "ROC_AUC"]

FEATURIZATIONS = ["morgan_fp", "mordred_desc"]

ORDERED_FRAMEWORKS = {
    framework: idx
    for idx, framework in enumerate(
        [
            "vnn",
            "autoadmet",
        ]
    )
}

ORDERED_FEATURIZATIONS = {
    featurization: idx
    for idx, featurization in enumerate(
        [
            "morgan_fp",
            "mordred_desc",
            "ensemble",
        ]
    )
}


def make_combined_cv_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    results_summary_dfs: list[pd.DataFrame] = []
    per_fold_results_dfs: list[pd.DataFrame] = []
    for dataset in datasets["DATASET"]:
        try:
            results_summary_file = (
                f"output/autoadmet_pipeline/{dataset}/cv_results/results_summary.csv"
            )
            results_summary = pd.read_csv(results_summary_file)
            results_summary.insert(0, "dataset", dataset)
            results_summary_dfs.append(results_summary)

            per_fold_results_file = f"output/autoadmet_pipeline/{dataset}/cv_results/per_fold_results.csv"
            per_fold_results = pd.read_csv(per_fold_results_file)
            per_fold_results.insert(0, "dataset", dataset)
            per_fold_results_dfs.append(per_fold_results)
        except Exception:
            print(f"No validation performance for {dataset}")

    results_summary: pd.DataFrame = pd.concat(results_summary_dfs).reset_index(drop=True)
    per_fold_results: pd.DataFrame = pd.concat(per_fold_results_dfs).reset_index(drop=True)

    results_summary.to_csv(
        "visualization/out/combined_cv_results_summary.csv",
        index=False,
    )

    per_fold_results.to_csv(
        "visualization/out/combined_cv_per_fold_results.csv",
        index=False,
    )
    return (results_summary, per_fold_results)


def main():
    Path("visualization/out").mkdir(exist_ok=True)

    # Make plot configurations
    plot_configs = [
        (
            "vnn",
            "morgan_fp",
            0,
            "vNN - Morgan Fingerprints",
            "#56B4E9",
        ),
        (
            "autoadmet",
            "morgan_fp",
            1,
            "AutoADMET Pipeline Best Predictor - Morgan Fingerprints",
            "#F0E442",
        ),
        (
            "autoadmet",
            "mordred_desc",
            1,
            "AutoADMET Pipeline Best Predictor - Mordred Descriptors",
            "#E69F00",
        ),
    ]
    plot_configs = {
        (framework, featurization): (
            color,
            SPREAD * (offset  - 0.5),
            display_name,
        )
        for framework, featurization, offset, display_name, color in plot_configs
    }

    results_summary, per_fold_results = make_combined_cv_dfs()

    # Get the per fold results for only the best validating AutoML models
    automl_best = (
        results_summary[results_summary["Model"] != "vNN"]
        .sort_values("Kappa", ascending=False)
        .drop_duplicates("dataset")
    )
    is_best_model = (
        per_fold_results[["dataset", "Descriptor", "Model"]].merge(
            automl_best[["dataset", "Descriptor", "Model"]], how="left", indicator=True
        )["_merge"]
        == "both"
    )
    best_model_per_fold_results = per_fold_results[is_best_model].copy()
    best_model_per_fold_results["framework"] = "autoadmet"

    # Get the per fold results for only the vNN models
    vnn_per_fold_results = per_fold_results[per_fold_results["Model"] == "vNN"].copy()
    vnn_per_fold_results["framework"] = "vnn"

    # Read in dataset config
    dataset_config = pd.read_csv("visualization/dataset_config.csv")
    dataset_config = dataset_config[["DATASET", "DISPLAY_NAME", "TRAIN_SIZE"]]
    dataset_config = dataset_config.sort_values("TRAIN_SIZE").reset_index(drop=True)

    # Combine the per fold results to plot
    plot_per_fold_results = pd.concat([vnn_per_fold_results, best_model_per_fold_results])
    desc_to_display = {
        "Morgan": "morgan_fp",
        "Mordred": "mordred_desc",
    }
    plot_per_fold_results["featurization"] = plot_per_fold_results["Descriptor"].replace(desc_to_display)

    for metric in METRICS:
        # Make the box plot
        fig = plt.figure(figsize=(12, 6), dpi=300)
        ax = fig.add_subplot()
        ax.set_ylim(bottom=0, top=1)
        ax.set_title(f"Validation {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        for _, framework, featurization in (
            plot_per_fold_results[["framework", "featurization"]].drop_duplicates().itertuples()
        ):
            color, offset, _ = plot_configs[(framework, featurization)]

            per_fold_results_subsection = plot_per_fold_results[
                (plot_per_fold_results["framework"] == framework)
                & (plot_per_fold_results["featurization"] == featurization)
            ]
            xs = [
                per_fold_results_subsection[per_fold_results_subsection["dataset"] == dataset][
                    metric
                ].to_numpy()
                for dataset in dataset_config["DATASET"]
            ]
            positions = dataset_config.index.to_numpy() + offset

            bplot = ax.boxplot(
                xs, positions=positions, patch_artist=True, widths=[0.3] * len(xs)
            )
            for box in bplot["boxes"]:
                box.set_facecolor(color)
            for median in bplot["medians"]:
                median.set_color("#000000")

        ax.xaxis.set_ticks(
            dataset_config.index.to_numpy(),
            dataset_config["DISPLAY_NAME"],
            rotation=45,
            ha="right",
        )

        handles = [
            (
                Patch(edgecolor="black", facecolor=color),
                display_name,
            )
            for (color, _, display_name) in plot_configs.values()
        ]
        ax.legend(*zip(*handles))
        
        fig.savefig(
            f"visualization/out/box_plots.{metric.lower()}.validation.png", bbox_inches="tight"
        )

if __name__ == "__main__":
    main()
