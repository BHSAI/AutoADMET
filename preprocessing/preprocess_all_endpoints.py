import preprocess_compounds

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
