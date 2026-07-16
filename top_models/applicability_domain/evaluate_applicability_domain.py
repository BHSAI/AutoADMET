import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from vnn import variable_nearest_neighbor as vnn_lib
import sklearn.metrics as metrics
import pandas as pd
import numpy as np

datasets = pd.read_csv("visualization/dataset_config.csv")
datasets = datasets.sort_values("FULL_SIZE").reset_index(drop=True)
SPREAD = 0.33
plot_configs = [
    (
        "vnn",
        "morgan_fp",
        0,
        "vNN - Morgan Fingerprints",
        "#56B4E9",
    ),
    (
        "bhsai_automl",
        "morgan_fp",
        1,
        "BHSAI AutoML Pipeline Best Predictor - Morgan Fingerprints",
        "#F0E442",
    ),
    (
        "bhsai_automl",
        "mordred_desc",
        1,
        "BHSAI AutoML Pipeline Best Predictor - Mordred Descriptors",
        "#E69F00",
    ),
    (
        "bhsai_automl",
        "ensemble",
        2,
        "BHSAI AutoML Pipeline Ensemble Predictor",
        "#009E73",
    ),
]
plot_configs = {
    (framework, featurization): (
        color,
        offset * SPREAD / (3 - 1) - SPREAD / 2,
        display_name,
    )
    for framework, featurization, offset, display_name, color in plot_configs
}
DESCIPTORS = {
    "Morgan": "morgan_fp",
    "Mordred": "mordred_desc",
}


def evaluate(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    **kwargs,
) -> dict:
    return {
        "kappa": metrics.cohen_kappa_score(y_test, y_pred),
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred, pos_label=1),
        "specificity": metrics.recall_score(y_test, y_pred, pos_label=0),
        **kwargs,
    }


def evaluate_with_applicability_domain(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_within_app_dom: np.ndarray,
) -> pd.DataFrame:
    app_dom_coverage = len(y_test[y_within_app_dom]) / len(y_pred)
    return pd.DataFrame(
        {
            "full": evaluate(
                y_test,
                y_pred,
                coverage=1,
            ),
            "inside_app_dom": evaluate(
                y_test[y_within_app_dom],
                y_pred[y_within_app_dom],
                coverage=app_dom_coverage,
            ),
            "outside_app_dom": evaluate(
                y_test[~y_within_app_dom],
                y_pred[~y_within_app_dom],
                coverage=1 - app_dom_coverage,
            ),
        }
    )


def get_y_test_and_applicability_domain(
    dataset: str,
    distance_threshold: float,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.floating]],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
]:
    train = pd.read_csv(f"data/preprocessed/{dataset}/morgan_fp.train.csv")
    X_train = train[[col for col in train.columns if "FEATURE_" in col]].to_numpy()
    y_train = train["CLASS"].to_numpy()

    test = pd.read_csv(f"data/preprocessed/{dataset}/morgan_fp.test.csv")
    X_test = test[[col for col in test.columns if "FEATURE_" in col]].to_numpy()
    y_test = test["CLASS"].to_numpy()

    restr_app_dom_predictor = vnn_lib.VariableNearestNeighborsClassifier(
        1, distance_threshold
    )
    restr_app_dom_predictor.fit(X_train, y_train)
    y_within_app_dom = ~np.isnan(restr_app_dom_predictor.predict(X_test))

    return y_test, y_within_app_dom


def get_metrics_for_dataset(
    dataset: str,
    distance_threshold: float,
) -> pd.DataFrame:
    y_test, y_within_app_dom = get_y_test_and_applicability_domain(
        dataset, distance_threshold
    )

    y_pred_vnn = np.load(f"top_models/vnn/y_pred.{dataset}.npy")
    vnn_metrics = evaluate_with_applicability_domain(
        y_test=y_test,
        y_pred=y_pred_vnn,
        y_within_app_dom=y_within_app_dom,
    )
    vnn_metrics.columns = pd.MultiIndex.from_product(
        [["vnn-morgan_fp"], vnn_metrics.columns],
    )

    y_pred_automl_best = pd.read_csv(
        f"output/bhsai_automl_pipeline/{dataset}/final_model/best_model_test_predictions.csv"
    )["Predicted_Label"].to_numpy()
    automl_best_metrics = evaluate_with_applicability_domain(
        y_test=y_test,
        y_pred=y_pred_automl_best,
        y_within_app_dom=y_within_app_dom,
    )
    with open(
        f"output/bhsai_automl_pipeline/{dataset}/final_model/model_metadata.json"
    ) as file:
        automl_best_descriptor = json.load(file)["descriptor"]
    automl_best_metrics.columns = pd.MultiIndex.from_product(
        [
            [f"bhsai_automl-{DESCIPTORS[automl_best_descriptor]}"],
            automl_best_metrics.columns,
        ],
    )

    y_pred_automl_ensemble = pd.read_csv(
        f"output/bhsai_automl_pipeline/{dataset}/final_model/ensemble_test_predictions.csv"
    )["Predicted_Label"].to_numpy()
    automl_ensemble_metrics = evaluate_with_applicability_domain(
        y_test=y_test,
        y_pred=y_pred_automl_ensemble,
        y_within_app_dom=y_within_app_dom,
    )
    automl_ensemble_metrics.columns = pd.MultiIndex.from_product(
        [["bhsai_automl-ensemble"], automl_ensemble_metrics.columns],
    )

    return pd.concat(
        [vnn_metrics, automl_best_metrics, automl_ensemble_metrics], axis=1
    )


def make_applicability_domain_performance_plot(df: pd.DataFrame):
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_subplot()
    ax.set_ylim(bottom=0, top=1)
    _ = ax.xaxis.set_ticks(
        datasets.index.to_numpy(),
        datasets["DISPLAY_NAME"],
        rotation=45,
        ha="right",
    )
    for dataset in datasets["DATASET"]:
        for config, column in df[dataset].items():
            if config != "coverage":
                framework, featurization = str.split(config, "-")
                full, inside, outside = column
                color, offset, _ = plot_configs[(framework, featurization)]  # type: ignore
                x = datasets[datasets["DATASET"] == dataset].index[0] + offset
                ax.plot(x, inside, "*", color=color)
                ax.plot(x, full, "o", color=color)
                ax.plot(x, outside, ".", color=color)
                ax.plot(
                    [x, x],
                    [column.min(), column.max()],
                    color=color,
                )

    handles = [
        (
            Line2D([], [], color="black", marker="*", linestyle="None"),
            "Inside applicability domain",
        ),
        (
            Line2D([], [], color="black", marker="o", linestyle="None"),
            "Full test set",
        ),
        (
            Line2D([], [], color="black", marker=".", linestyle="None"),
            "Outside applicability domain",
        ),
    ]
    marker_legend = ax.legend(*zip(*handles), loc="upper left")
    ax.add_artist(marker_legend)
    handles = [
        (
            Line2D([0], [0], color=color),
            display_name,
        )
        for (color, _, display_name) in plot_configs.values()  # type: ignore
    ]
    ax.legend(*zip(*handles), loc="upper right")
    ax.set_title("Performance on vNN applicability domain")
    ax.set_xlabel("Model")
    ax.set_ylabel("Kappa")
    fig.savefig("visualization/out/applicability_domain_kappa.png", bbox_inches="tight")


def main():
    METRIC = "kappa"
    performance_summaries = []
    for _, dataset, distance_threshold in datasets[
        ["DATASET", "DISTANCE_THRESHOLD"]
    ].itertuples():
        evaluation = get_metrics_for_dataset(dataset, distance_threshold)
        evaluation.to_csv(
            f"top_models/applicability_domain/applicability_domain_metrics.{dataset}.csv"
        )

        kappa = evaluation.loc[METRIC].unstack().T
        kappa.columns = pd.MultiIndex.from_product([[dataset], kappa.columns])  # type: ignore
        coverage = evaluation.loc["coverage"]["vnn-morgan_fp"].to_frame()
        coverage.columns = pd.MultiIndex.from_product([[dataset], coverage.columns])

        performance_summaries.append(pd.concat([kappa, coverage], axis=1))

    performance_summary = pd.concat(performance_summaries, axis=1)
    performance_summary.T.to_csv(
        f"top_models/applicability_domain/applicability_domain_{METRIC}_summary.csv"
    )

    make_applicability_domain_performance_plot(performance_summary)


if __name__ == "__main__":
    main()
