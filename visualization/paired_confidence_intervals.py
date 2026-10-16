import itertools as it
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

DATASETS = pd.read_csv(Path(__file__).parents[1] / "data" / "dataset_config.csv")[
    "DATASET"
].to_list()
FRAMEWORK_ORDER = {
    framework: idx
    for idx, framework in enumerate(
        [
            "vnn-morgan_fp",
            "autoadmet-single_model",
            "autoadmet-morgan_fp",
            "autoadmet-mordred_desc",
            "autoadmet-ensemble",
            "autogluon-morgan_fp",
            "autogluon-mordred_desc",
            "flaml-morgan_fp",
            "flaml-mordred_desc",
        ]
    )
}
USE_EXTRA_FRAMEWORKS = True


def load_predictions_slice(
    filename: str | Path,
    smiles_col: str = "SMILES",
    target_smiles_col_name: str = "SMILES",
    class_col: str = "CLASS",
    target_class_col_name: str = "CLASS",
) -> pd.DataFrame:
    df = pd.read_csv(filename)[[smiles_col, class_col]]
    df = df.rename(
        axis=1,
        mapper={
            smiles_col: target_smiles_col_name,
            class_col: target_class_col_name,
        },
    )
    return df


def load_predictions(dataset: str):
    pred_by_framework = load_predictions_slice(
        filename=f"data/preprocessed/{dataset}/mordred_desc.test.csv",
        target_class_col_name="y_true",
    )

    config_to_pred_file = {
        "autoadmet-best_model": f"output/autoadmet_pipeline/{dataset}/final_model/best_model_test_predictions.csv",
        "autoadmet-ensemble": f"output/autoadmet_pipeline/{dataset}/final_model/ensemble_test_predictions.csv",
        "vnn-morgan_fp": f"output/vnn/predictions/predictions.{dataset}.csv",
    }

    if USE_EXTRA_FRAMEWORKS:
        config_to_pred_file.update(
            {
                "autogluon-mordred_desc": f"output/autogluon/predictions/predictions.{dataset}.mordred_desc.quadratic_kappa.no_time_limit.csv",
                "autogluon-morgan_fp": f"output/autogluon/predictions/predictions.{dataset}.morgan_fp.quadratic_kappa.no_time_limit.csv",
                "flaml-mordred_desc": f"output/flaml/predictions/predictions.{dataset}.mordred_desc.75min.csv",
                "flaml-morgan_fp": f"output/flaml/predictions/predictions.{dataset}.morgan_fp.75min.csv",
            }
        )

    for configuration, predictions_file in config_to_pred_file.items():
        pred_by_framework = pred_by_framework.merge(
            right=load_predictions_slice(
                filename=predictions_file,
                class_col="Predicted_Label",
                target_class_col_name=configuration,
            ),
            on="SMILES",
        )

    return pred_by_framework


def kappa_difference(
    y_true: np.ndarray,
    pred_1: np.ndarray,
    pred_2: np.ndarray,
):
    return metrics.cohen_kappa_score(
        y_true,
        pred_1,
    ) - metrics.cohen_kappa_score(
        y_true,
        pred_2,
    )


def main():
    # x = {
    #     "autogluon-morgan_fp": {
    #         ("autogluon-mordred_desc", "lower"): 1,
    #         ("autogluon-mordred_desc", "mean"): 2,
    #         ("autogluon-mordred_desc", "upper"): 3,
    #         ("flaml-mordred_desc", "lower"): 1,
    #         ("flaml-mordred_desc", "mean"): 2,
    #         ("flaml-mordred_desc", "upper"): 3,
    #     }
    # }
    out_dir = Path("output/pairwise_comparisons")
    out_dir.mkdir(exist_ok=True, parents=True)

    for dataset in DATASETS:
        print(f"Working on {dataset}")
        pairwise_comparisons: dict[
            str,
            dict[tuple[str, str], float],
        ] = defaultdict(lambda: {})

        pred_by_framework = load_predictions(dataset)

        y_true = pred_by_framework["y_true"].to_numpy()
        frameworks = pred_by_framework.columns[2:]

        for config_1, config_2 in it.combinations(frameworks, r=2):
            print(f"\t Getting bootstrap CI between {config_1} and {config_2}")

            pred_1 = pred_by_framework[config_1].to_numpy()
            pred_2 = pred_by_framework[config_2].to_numpy()

            diff_kappa = kappa_difference(y_true, pred_1, pred_2)

            low, high = stats.bootstrap(
                (y_true, pred_1, pred_2),
                statistic=kappa_difference,
                confidence_level=0.95,
                paired=True,
            ).confidence_interval

            pairwise_comparisons[config_1].update(
                {
                    (config_2, "low"): low,
                    (config_2, "mean"): diff_kappa,
                    (config_2, "high"): high,
                }
            )

            pairwise_comparisons[config_2].update(
                {
                    (config_1, "low"): -low,
                    (config_1, "mean"): -diff_kappa,
                    (config_1, "high"): -high,
                }
            )

        df = pd.DataFrame.from_dict(pairwise_comparisons)
        df = df.sort_index(axis=0, key=lambda idx: idx.map(FRAMEWORK_ORDER))
        df = df.sort_index(axis=1, key=lambda idx: idx.map(FRAMEWORK_ORDER))
        df.to_csv(out_dir / f"{dataset}.csv")


if __name__ == "__main__":
    main()
