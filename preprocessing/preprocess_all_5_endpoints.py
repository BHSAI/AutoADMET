import preprocess_compounds

DATASETS = [
    "ames",
    "cytotox",
    "dili",
    "hlm",
    "mmp",
]


def main():
    for featurization in preprocess_compounds.featurizers.keys():
        for dataset in DATASETS:
            preprocess_compounds.preprocess_compounds(
                raw_data_csv=f"./data/raw/{dataset}.csv",
                output_dir=f"./data/preprocessed/{dataset}",
                class_col="Property",
                featurization=featurization,
            )


if __name__ == "__main__":
    main()
