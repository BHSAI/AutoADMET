import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import rdBase

import preprocess_compounds

DATASETS = pd.read_csv(Path(__file__).parents[1] / "data" / "dataset_config.csv")


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    preprocessing_time = defaultdict(lambda: {})
    for dataset in DATASETS["DATASET"]:
        logger.info(f"Generating features for {dataset}")
        with rdBase.BlockLogs():
            _, cleaning_time, featurization_times = (
                preprocess_compounds.preprocess_compounds(
                    raw_data_csv=f"./data/raw/{dataset}.csv",
                    output_dir=f"./data/preprocessed/{dataset}",
                    class_col="Property",
                    featurizations=list(preprocess_compounds.featurizers.keys()),
                )
            )
        logger.info(f"Finished features for {dataset}")
        logger.info(f"Cleaning time: {cleaning_time:.2f}")

        preprocessing_time["cleaning_time"][dataset] = cleaning_time
        for featurization, time in featurization_times.items():
            logger.info(f"Time for {featurization}: {time:.2f}")
            preprocessing_time[featurization][dataset] = time

    #  Make table of preprocessing times
    preprocessing_time_df = pd.DataFrame.from_dict(preprocessing_time)

    # Get normalized times
    time_columns = list(preprocessing_time_df.columns)
    preprocessing_time_df = preprocessing_time_df.merge(
        right=DATASETS[["DATASET", "FULL_SIZE"]].set_index("DATASET", drop=True),
        left_index=True,
        right_index=True,
    )
    for col in time_columns:
        preprocessing_time_df[f"{col}-normalized"] = (
            preprocessing_time_df[col] / preprocessing_time_df["FULL_SIZE"]
        )

    # Save
    preprocessing_time_df.to_csv("data/preprocessing_time.csv")


if __name__ == "__main__":
    main()
