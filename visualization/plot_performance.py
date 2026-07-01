from pathlib import Path

import pandas as pd
import itertools as it
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from pathlib import Path
from collections import defaultdict
import adjustText

datasets = pd.read_csv(Path(__file__).parent / "dataset_config.csv")
datasets = datasets.sort_values("FULL_SIZE").reset_index(drop=True)

SPREAD = 0.33
TICK_WIDTH = 0.08
METRICS = ["kappa", "accuracy", "recall", "specificity"]

FEATURIZATIONS = ["morgan_fp", "mordred_desc"]

ORDERED_FRAMEWORKS = {
    framework: idx
    for idx, framework in enumerate(
        [
            "vnn",
            "flaml",
            "autogluon",
            "bhsai_automl",
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


def make_combined_performance_df() -> pd.DataFrame:
    vnn_performances: list[pd.DataFrame] = []
    for dataset in datasets["DATASET"]:
        try:
            filename = f"top_models/vnn/top_model.{dataset}.morgan_fp.csv"
            df = pd.read_csv(filename)
            df.insert(0, "featurization", "morgan_fp")
            df.insert(0, "dataset", dataset)
            df.insert(0, "framework", "vnn")
            vnn_performances.append(df)
        except Exception:
            print(f"No vNN performance for {dataset}")

    bhsai_pipeline_performances: list[pd.DataFrame] = []
    for dataset in datasets["DATASET"]:
        try:
            filename = f"top_models/bhsai_automl_pipeline/top_model.{dataset}.csv"
            df = pd.read_csv(filename)
            top_featurization, _ = str.split(df["details"][0], "-")
            top_featurization = {"Morgan": "morgan_fp", "Mordred": "mordred_desc"}[
                top_featurization
            ]
            df.insert(0, "featurization", [top_featurization, "ensemble"])
            df.insert(0, "dataset", dataset)
            df.insert(0, "framework", "bhsai_automl")
            bhsai_pipeline_performances.append(df)
        except Exception:
            print(f"No BHSAI internal pipeline performance for {dataset}")

    combined_performance_df: pd.DataFrame = pd.concat(
        [
            *vnn_performances,
            *bhsai_pipeline_performances,
        ]
    )

    # Add normalized training time
    combined_performance_df = combined_performance_df.reset_index(drop=True)
    df = combined_performance_df.merge(datasets, left_on="dataset", right_on="DATASET")
    combined_performance_df["train_time_normalized"] = (
        df["train_time"] / df["TRAIN_SIZE"]
    )

    combined_performance_df.to_csv(
        "visualization/out/combined_performance.csv",
        index=False,
    )
    return combined_performance_df


def main():
    Path("visualization/out").mkdir(exist_ok=True)

    combined_performance_df = make_combined_performance_df()

    # Make plot configurations
    fit_configs = [
        ("vnn", "morgan_fp", 0, "vNN - Morgan Fingerprints"),
        (
            "bhsai_automl",
            "morgan_fp",
            1,
            "BHSAI AutoML Pipeline Best Predictor - Morgan Fingerprints",
        ),
        (
            "bhsai_automl",
            "mordred_desc",
            1,
            "BHSAI AutoML Pipeline Best Predictor - Mordred Descriptors",
        ),
        ("bhsai_automl", "ensemble", 2, "BHSAI AutoML Pipeline Ensemble Predictor"),
    ]
    num_configs = len(fit_configs)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors = colors * int(1 + num_configs / len(colors))
    plot_configs = {
        (framework, featurization): (
            colors[i],
            offset * SPREAD / (3 - 1) - SPREAD / 2,
            display_name,
        )
        for i, (framework, featurization, offset, display_name) in enumerate(
            fit_configs
        )
    }

    #  Plot test performance
    for metric in METRICS:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        ax.set_ylim(bottom=0, top=1)
        ax.xaxis.set_ticks(
            datasets.index.to_numpy(), datasets["DISPLAY_NAME"], rotation=45, ha="right"
        )
        ax.set_title(f"Test {metric.capitalize()} 95% Confidence Interval")
        ax.set_xlabel("Dataset")
        for _, row in combined_performance_df.iterrows():
            dataset, framework, featurization = row[
                ["dataset", "framework", "featurization"]
            ]
            bottom, center, top = row[[f"{metric}-lb", metric, f"{metric}-ub"]]
            color, offset, _ = plot_configs[(framework, featurization)]
            x = datasets[datasets["DATASET"] == dataset].index[0] + offset
            left = x - TICK_WIDTH / 2
            right = x + TICK_WIDTH / 2

            ax.plot([x, x], [top, bottom], color=color)
            ax.plot([left, right], [top, top], color=color)
            ax.plot([left, right], [bottom, bottom], color=color)
            ax.plot(x, center, "o", color=color)

        handles = [
            (
                Line2D([0], [0], color=color),
                display_name,
            )
            for (color, _, display_name) in plot_configs.values()
        ]
        ax.legend(*zip(*handles))

        fig.savefig(
            f"visualization/out/confidence_intervals.{metric}.png", bbox_inches="tight"
        )

    #  Plot time test performance
    for time_key, time_display_name, unit, multiplier in [
        ("train_time_normalized", "Training time, per 1000 compounds", "h", 1 / 3600),
        (
            "pred_time_normalized",
            "Inference time on test set, per 1,000 compounds",
            "s",
            1,
        ),
        (
            "pred_time_representative-20000_normalized",
            "Inference time on 20,000 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
        ),
        (
            "pred_time_representative-10000_normalized",
            "Inference time on 10,000 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
        ),
        (
            "pred_time_representative-5000_normalized",
            "Inference time on 5,000 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
        ),
        (
            "pred_time_representative-2500_normalized",
            "Inference time on 2,500 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
        ),
        (
            "pred_time_small_compounds_normalized",
            "Inference time on small compounds, per 1,000 compounds",
            "s",
            1,
        ),
        (
            "pred_time_large_compounds_normalized",
            "Inference time on large compounds, per 1,000 compounds",
            "s",
            1,
        ),
    ]:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot()
        ax.xaxis.set_ticks(
            datasets.index.to_numpy(), datasets["DISPLAY_NAME"], rotation=45, ha="right"
        )
        ax.set_title(time_display_name)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(f"Normalized time ({unit})")
        # ax.set_ylim(bottom=0, top=3.1)
        for _, row in combined_performance_df.iterrows():
            dataset, framework, featurization = row[
                ["dataset", "framework", "featurization"]
            ]
            y = row[time_key] * 1000 * multiplier
            color, offset, _ = plot_configs[(framework, featurization)]
            x = datasets[datasets["DATASET"] == dataset].index[0] + offset

            ax.bar(x, height=y, width=0.1, color=color)

        handles = [
            (
                Line2D([0], [0], color=color),
                display_name,
            )
            for (color, _, display_name) in plot_configs.values()
        ]
        ax.legend(*zip(*handles))

        fig.savefig(f"visualization/out/{time_key}.png", bbox_inches="tight")

    # Make stacked bar chart
    combined_performance_df["config"] = combined_performance_df.apply(
        lambda row: plot_configs[(row["framework"], row["featurization"])][2],
        axis=1,
    )
    combined_performance_df["config"] = combined_performance_df["config"].apply(
        lambda config: str(config)
        .removesuffix(" - Morgan Fingerprints")
        .removesuffix(" - Mordred Descriptors")
    )
    ranking = (
        combined_performance_df[["config", "dataset", "kappa"]]
        .sort_values("kappa", ascending=False)
        .sort_values("dataset", kind="stable")
    )

    # Count ranks for each configuration
    ranking_counts = defaultdict(lambda: defaultdict(lambda: 0))
    num_configs = ranking["config"].drop_duplicates().shape[0]
    unique_datasets = ranking["dataset"].drop_duplicates()
    for dataset in unique_datasets:
        dataset_ranking = ranking[ranking["dataset"] == dataset].reset_index(drop=True)
        for i in range(num_configs):
            ranking_counts[dataset_ranking["config"][i]][i + 1] += 1

    # Plot stacked bar chart
    ax = pd.DataFrame(ranking_counts).fillna(0).plot.bar(stacked=True)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Number of Models")
    ax.set_xticklabels(["1st", "2nd", "3rd"])
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Models")
    ax.figure.savefig(f"visualization/out/ranking_stacked_bar.png", bbox_inches="tight")  # type: ignore

    # Make inference time scatter plot
    for index, name in [
        (combined_performance_df["framework"] == "vnn", "vNN"),
        (
            (combined_performance_df["framework"] == "bhsai_automl")
            & (combined_performance_df["featurization"] != "ensemble"),
            "BHSAI AutoML best predictor",
        ),
        (
            (combined_performance_df["framework"] == "bhsai_automl")
            & (combined_performance_df["featurization"] == "ensemble"),
            "BHSAI AutoML ensemble",
        ),
    ]:
        df = combined_performance_df[index].copy()
        df = df.merge(
            datasets[["DATASET", "TRAIN_SIZE", "DISPLAY_NAME"]],
            left_on="dataset",
            right_on="DATASET",
        )
        df["pred_time_representative-20000_normalized"] = (
            df["pred_time_representative-20000_normalized"] * 1000
        )
        ax = df.plot.scatter(
            x="TRAIN_SIZE", y="pred_time_representative-20000_normalized", label="Dataset"
        )
        ax.set_xlabel("Number of training compounds")
        ax.set_ylabel("Inference time per 1000 compounds (s)")
        ax.set_title(f"Effect of training set size on {name} inference cost")
        texts = [
            ax.text(x, y, dataset, fontsize=8)
            for _, dataset, x, y in df[
                ["DISPLAY_NAME", "TRAIN_SIZE", "pred_time_representative-20000_normalized"]
            ].itertuples()
        ]
        adjustText.adjust_text(
            texts,
            expand=(2, 2),
            arrowprops=dict(arrowstyle="-", lw=1),
            min_arrow_len=0,
            force_static=(1, 1),
            ax=ax,
            color="gray",
        )
        ax.figure.savefig(f"visualization/out/inference_time_scatter_plot.{name.lower().replace(" ", "_")}.png", bbox_inches="tight")  # type: ignore


if __name__ == "__main__":
    main()
