import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import rdBase

import preprocess_compounds

DATASETS = pd.read_csv(Path(__file__).parents[1] / "data" / "dataset_config.csv")[
    "DATASET"
].to_list()


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    preprocessing_time = defaultdict(lambda: {})
    for dataset in DATASETS:
        logger.info(f"Generating features for {dataset}")
        with rdBase.BlockLogs():
            _, cleaning_time, featurization_times = preprocess_compounds.preprocess_compounds(
                raw_data_csv=f"./data/raw/{dataset}.csv",
                output_dir=f"./data/preprocessed/{dataset}",
                class_col="Property",
                featurizations=list(preprocess_compounds.featurizers.keys()),
            )
        logger.info(f"Finished features for {dataset}")
        logger.info(f"Cleaning time: {cleaning_time:.2f}")

        preprocessing_time["cleaning_time"][dataset] = cleaning_time
        for featurization, time in featurization_times.items():
            logger.info(f"Time for {featurization}: {time:.2f}")
            preprocessing_time[featurization][dataset] = time


    pd.DataFrame.from_dict(preprocessing_time).to_csv("data/preprocessing_time.csv")

if __name__ == "__main__":
    main()
