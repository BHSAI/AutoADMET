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

datasets = pd.read_csv(Path(__file__).parent.parent / "data" / "dataset_config.csv")
datasets = datasets.sort_values("FULL_SIZE").reset_index(drop=True)

USE_EXTRA_FRAMEWORKS = False
SPREAD = 0.75 if USE_EXTRA_FRAMEWORKS else 0.33
TICK_WIDTH = 0.08
METRICS = ["kappa", "accuracy", "recall", "specificity"]

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


# Make plot configurations
PLOT_CONFIGS = [
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
    (
        "autoadmet",
        "ensemble",
        2,
        "AutoADMET Pipeline Ensemble Predictor",
        "#009E73",
    ),
]

PLOT_CONFIGS_EXTRA_FRAMEWORKS = [
    (
        "autogluon",
        "morgan_fp",
        3,
        "AutoGluon - Morgan Fingerprints",
        "#D55E00",
    ),
    (
        "autogluon",
        "mordred_desc",
        4,
        "AutoGluon - Mordred Descriptors",
        "#CC79A7",
    ),
    (
        "flaml",
        "morgan_fp",
        5,
        "FLAML - Morgan Fingerprints",
        "#0072B2",
    ),
    (
        "flaml",
        "mordred_desc",
        6,
        "FLAML - Mordred Descriptors",
        "#000000",
    ),
]


def make_combined_performance_df() -> pd.DataFrame:
    performance_dfs: list[pd.DataFrame] = []

    for dataset in datasets["DATASET"]:
        try:
            filename = f"top_models/vnn/top_model.{dataset}.morgan_fp.csv"
            df = pd.read_csv(filename)
            df.insert(0, "featurization", "morgan_fp")
            df.insert(0, "dataset", dataset)
            df.insert(0, "framework", "vnn")
            performance_dfs.append(df)
        except Exception:
            print(f"No vNN performance for {dataset}")

    for dataset in datasets["DATASET"]:
        try:
            filename = f"top_models/autoadmet_pipeline/top_model.{dataset}.csv"
            df = pd.read_csv(filename)
            top_featurization, _ = str.split(df["details"][0], "-")
            top_featurization = {"Morgan": "morgan_fp", "Mordred": "mordred_desc"}[
                top_featurization
            ]
            df.insert(0, "featurization", [top_featurization, "ensemble"])
            df.insert(0, "dataset", dataset)
            df.insert(0, "framework", "autoadmet")
            performance_dfs.append(df)
        except Exception:
            print(f"No BHSAI internal pipeline performance for {dataset}")

    if USE_EXTRA_FRAMEWORKS:
        for dataset, featurization in it.product(datasets["DATASET"], FEATURIZATIONS):
            try:
                filename = f"top_models/autogluon/top_model.{dataset}.{featurization}.quadratic_kappa.no_time_limit.csv"
                df = pd.read_csv(filename)
                df.insert(0, "featurization", featurization)
                df.insert(0, "dataset", dataset)
                df.insert(0, "framework", "autogluon")
                performance_dfs.append(df)
            except Exception:
                print(f"No AutoGluon performance for {dataset}, {featurization}")

        for dataset, featurization in it.product(datasets["DATASET"], FEATURIZATIONS):
            try:
                filename = (
                    f"top_models/flaml/top_model.{dataset}.{featurization}.75min.csv"
                )
                df = pd.read_csv(filename)
                df.insert(0, "featurization", featurization)
                df.insert(0, "dataset", dataset)
                df.insert(0, "framework", "flaml")
                performance_dfs.append(df)
            except Exception:
                print(f"No FLAML performance for {dataset}, {featurization}")

    combined_performance_df: pd.DataFrame = pd.concat(performance_dfs)

    # Add normalized training time
    combined_performance_df = combined_performance_df.reset_index(drop=True)
    df = combined_performance_df.merge(datasets, left_on="dataset", right_on="DATASET")
    combined_performance_df["train_time_normalized"] = (
        df["train_time"] / df["TRAIN_SIZE"]
    )

    combined_performance_df.to_csv(
        f"visualization/out/combined_performance{"-extra_frameworks" if USE_EXTRA_FRAMEWORKS else ""}.csv",
        index=False,
    )
    return combined_performance_df


def main():
    Path("visualization/out").mkdir(exist_ok=True)

    combined_performance_df = make_combined_performance_df()

    plot_configs_to_use = [
        *PLOT_CONFIGS,
        *(PLOT_CONFIGS_EXTRA_FRAMEWORKS if USE_EXTRA_FRAMEWORKS else []),
    ]

    n_offsets = len({offset for _, _, offset, _, _ in plot_configs_to_use})
    plot_configs = {
        (framework, featurization): (
            color,
            offset * SPREAD / (n_offsets - 1) - SPREAD / 2,
            display_name,
        )
        for framework, featurization, offset, display_name, color in plot_configs_to_use
    }

    combined_performance_df = combined_performance_df[
        combined_performance_df.apply(
            lambda row: (row["framework"], row["featurization"]) in plot_configs,
            axis=1,
        )
    ]

    #  Plot test performance
    for metric in METRICS:
        fig = plt.figure(figsize=(12, 6), dpi=300)
        ax = fig.add_subplot()
        ax.set_ylim(bottom=0, top=1)
        ax.xaxis.set_ticks(
            datasets.index.to_numpy(),
            datasets["DISPLAY_NAME"],
            rotation=45,
            ha="right",
        )
        ax.set_title(f"Test {metric.capitalize()} 95% Confidence Interval")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.capitalize())
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
    for time_key, time_display_name, unit, multiplier, y_max in [
        (
            "train_time_normalized",
            "Training time, per 1000 compounds",
            "h",
            1 / 3600,
            None,
        ),
        (
            "pred_time_normalized",
            "Inference time on test set, per 1,000 compounds",
            "s",
            1,
            None,
        ),
        (
            "pred_time_representative-20000_normalized",
            "Inference time on 20,000 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
            0.55,
        ),
        (
            "pred_time_representative-10000_normalized",
            "Inference time on 10,000 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
            0.55,
        ),
        (
            "pred_time_representative-5000_normalized",
            "Inference time on 5,000 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
            0.55,
        ),
        (
            "pred_time_representative-2500_normalized",
            "Inference time on 2,500 compound sample from all datasets, per 1,000 compounds",
            "s",
            1,
            0.55,
        ),
        (
            "pred_time_small_compounds_normalized",
            "Inference time on small compounds, per 1,000 compounds",
            "s",
            1,
            0.55,
        ),
        (
            "pred_time_large_compounds_normalized",
            "Inference time on large compounds, per 1,000 compounds",
            "s",
            1,
            0.55,
        ),
    ]:
        fig = plt.figure(figsize=(12, 6), dpi=300)
        ax = fig.add_subplot()
        ax.xaxis.set_ticks(
            datasets.index.to_numpy(), datasets["DISPLAY_NAME"], rotation=45, ha="right"
        )
        ax.set_title(time_display_name)
        ax.set_xlabel("Model")
        ax.set_ylabel(f"Normalized time ({unit})")
        if y_max:
            ax.set_ylim(bottom=0, top=y_max)
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
    ranking = (
        combined_performance_df[["config", "dataset", "kappa"]]
        .sort_values("kappa", ascending=False)
        .sort_values("dataset", kind="stable")
    )

    # Count ranks for each configuration
    ranking_counts = defaultdict(lambda: defaultdict(lambda: 0))
    unique_datasets = ranking["dataset"].drop_duplicates()
    for dataset in unique_datasets:
        dataset_ranking = ranking[ranking["dataset"] == dataset].reset_index(drop=True)
        for i in range(3):
            ranking_counts[dataset_ranking["config"][i]][i + 1] += 1

    # Plot stacked bar chart
    ax = (
        pd.DataFrame(ranking_counts)
        .sort_index(axis=1)
        .fillna(0)
        .plot.bar(
            stacked=True,
            color={
                display_name: color
                for (color, _, display_name) in plot_configs.values()
            },
        )
    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Number of Models")
    ax.set_xticklabels(["1st", "2nd", "3rd"])
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Models")
    ax.figure.savefig(  # type: ignore
        f"visualization/out/ranking_stacked_bar.png",
        bbox_inches="tight",
        dpi=300,
    )

    # Make inference time scatter plot
    for index, name in [
        (combined_performance_df["framework"] == "vnn", "vNN"),
        (
            (combined_performance_df["framework"] == "autoadmet")
            & (combined_performance_df["featurization"] != "ensemble"),
            "AutoADMET best predictor",
        ),
        (
            (combined_performance_df["framework"] == "autoadmet")
            & (combined_performance_df["featurization"] == "ensemble"),
            "AutoADMET ensemble",
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
            x="TRAIN_SIZE",
            y="pred_time_representative-20000_normalized",
            label="Model",
        )
        ax.set_xlabel("Number of training compounds")
        ax.set_ylabel("Inference time per 1000 compounds (s)")
        ax.set_title(f"Effect of training set size on {name} inference cost")
        texts = [
            ax.text(x, y, dataset, fontsize=8)
            for _, dataset, x, y in df[
                [
                    "DISPLAY_NAME",
                    "TRAIN_SIZE",
                    "pred_time_representative-20000_normalized",
                ]
            ].itertuples()
        ]
        adjustText.adjust_text(
            texts,
            expand=(2, 2),
            arrowprops=dict(arrowstyle="-", lw=1),
            min_arrow_len=0,
            ax=ax,
            color="gray",
        )
        ax.figure.savefig(  # type: ignore
            f"visualization/out/inference_time_scatter_plot.{name.lower().replace(" ", "_")}.png",
            bbox_inches="tight",
            dpi=300,
        )


if __name__ == "__main__":
    main()
