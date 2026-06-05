from pathlib import Path

import pandas as pd
import itertools as it
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

DATASETS = [
    "ames",
    "bbb",
    "cyp1a2",
    "cyp2c9",
    "cyp2c19",
    "cyp2d6",
    "cyp3a4",
    "cytotox",
    "dili",
    "herg",
    "hlm",
    "mmp",
    "pgp_inhibitors",
    "pgp_substrates",
]
# DATASETS_FMT = ["AMES", "Cytotox", "DILI", "HLM", "MMP"]
SPREAD = 0.5
TICK_WIDTH = 0.05
METRICS = ["kappa", "accuracy", "recall", "specificity"]
dataset_to_tick = {dataset.lower(): idx + 1 for idx, dataset in enumerate(DATASETS)}

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


def main():
    Path("visualization/out").mkdir(exist_ok=True)

    vnn_performances: list[pd.DataFrame] = []
    for dataset in DATASETS:
        try:
            filename = f"top_models/vnn/top_model.{dataset}.morgan_fp.csv"
            df = pd.read_csv(filename)
            df.insert(0, "featurization", "morgan_fp")
            df.insert(0, "dataset", dataset)
            df.insert(0, "framework", "vnn")
            vnn_performances.append(df)
        except Exception:
            print(f"No vNN performance for {dataset}")

    autogluon_performances: list[pd.DataFrame] = []
    for dataset, featurization in it.product(DATASETS, FEATURIZATIONS):
        try:
            filename = f"top_models/autogluon/top_model.{dataset}.{featurization}.quadratic_kappa.no_time_limit.csv"
            df = pd.read_csv(filename)
            df.insert(0, "featurization", featurization)
            df.insert(0, "dataset", dataset)
            df.insert(0, "framework", "autogluon")
            autogluon_performances.append(df)
        except Exception:
            print(f"No AutoGluon performance for {dataset}, {featurization}")

    flaml_performances: list[pd.DataFrame] = []
    for dataset, featurization in it.product(DATASETS, FEATURIZATIONS):
        try:
            filename = f"top_models/flaml/top_model.{dataset}.{featurization}.75min.csv"
            df = pd.read_csv(filename)
            df.insert(0, "featurization", featurization)
            df.insert(0, "dataset", dataset)
            df.insert(0, "framework", "flaml")
            flaml_performances.append(df)
        except Exception:
            print(f"No FLAML performance for {dataset}, {featurization}")

    bhsai_pipeline_performances: list[pd.DataFrame] = []
    for dataset in DATASETS:
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
            # *autogluon_performances,
            # *flaml_performances,
            *bhsai_pipeline_performances,
        ]
    )
    combined_performance_df.to_csv(
        "visualization/out/combined_performance.csv", index=False
    )

    fit_configs = [
        tuple(row[1:])
        for row in combined_performance_df[["framework", "featurization"]]
        .sort_values(
            "featurization",
            key=lambda row: row.apply(
                lambda featurization: ORDERED_FEATURIZATIONS[featurization]
            ),
        )
        .sort_values(
            "framework",
            key=lambda row: row.apply(lambda framework: ORDERED_FRAMEWORKS[framework]),
        )
        .drop_duplicates()
        .itertuples()
    ]
    num_configs = len(fit_configs)
    colors = list(mcolors.TABLEAU_COLORS.keys())
    colors = colors * int(1 + num_configs / len(colors))
    offsets = np.array(range(num_configs)) * SPREAD / (num_configs - 1) - SPREAD / 2
    plot_configs = dict(zip(fit_configs, zip(colors, offsets)))

    for metric in METRICS:
        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot()
        ax.set_ylim(bottom=0, top=1)
        ax.xaxis.set_ticks(list(dataset_to_tick.values()), DATASETS)
        ax.set_title(metric)
        for _, row in combined_performance_df.iterrows():
            dataset, framework, featurization = row[
                ["dataset", "framework", "featurization"]
            ]
            bottom, center, top = row[[f"{metric}-lb", metric, f"{metric}-ub"]]
            color, offset = plot_configs[(framework, featurization)]
            x = dataset_to_tick[dataset] + offset
            left = x - TICK_WIDTH / 2
            right = x + TICK_WIDTH / 2

            ax.plot([x, x], [top, bottom], color=color)
            ax.plot([left, right], [top, top], color=color)
            ax.plot([left, right], [bottom, bottom], color=color)
            ax.plot(x, center, "o", color=color)
            if metric == "kappa":
                ax.plot(x, row["kappa_val"], "x", color=color)

        handles = [
            (
                Line2D([0], [0], color=color),
                f"{dataset} - {featurization}",
            )
            for (dataset, featurization), (color, _) in plot_configs.items()
        ]
        if metric == "kappa":
            handles.append(
                (
                    Line2D([0], [0], color="black", marker="x", linestyle="None"),
                    "validation kappa",
                )
            )
        ax.legend(*zip(*handles), bbox_to_anchor=(1, 1), loc="upper left")

        fig.savefig(
            f"visualization/out/confidence_intervals.{metric}.png", bbox_inches="tight"
        )

    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot()
    ax.xaxis.set_ticks(list(dataset_to_tick.values()), DATASETS)
    ax.set_title("Search Time")
    for _, row in combined_performance_df.iterrows():
        dataset, framework, featurization = row[
            ["dataset", "framework", "featurization"]
        ]
        y = row["search_time"]
        color, offset = plot_configs[(framework, featurization)]
        x = dataset_to_tick[dataset] + offset

        ax.bar(x, height=y, width=0.1, color=color)

    handles = [
        (
            Line2D([0], [0], color=color),
            f"{dataset} - {featurization}",
        )
        for (dataset, featurization), (color, _) in plot_configs.items()
    ]
    ax.legend(*zip(*handles), bbox_to_anchor=(1, 1), loc="upper left")

    fig.savefig(f"visualization/out/search_time.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
