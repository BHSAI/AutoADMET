from pathlib import Path

import preprocess_compounds
import pandas as pd

datasets = pd.read_csv(Path(__file__).parent / "data" / "dataset_config.csv")[
    "DATASET"
].to_list()


def main():
    for featurization in preprocess_compounds.featurizers.keys():
        for dataset in datasets:
            preprocess_compounds.preprocess_compounds(
                raw_data_csv=f"./data/raw/{dataset}.csv",
                output_dir=f"./data/preprocessed/{dataset}",
                class_col="Property",
                featurization=featurization,
            )


if __name__ == "__main__":
    main()
