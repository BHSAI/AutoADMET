from pathlib import Path

import pandas as pd
import math

FEATURIZATIONS = ["morgan_fp", "mordred_desc", "canonical_smiles"]
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
SEED = 1031


def main():
    for featurization in FEATURIZATIONS:
        df = pd.concat(
            [
                pd.read_csv(f"data/preprocessed/{dataset}/{featurization}.all.csv")
                .sample(1500, replace=True, random_state=SEED)
                .reset_index(drop=True)
                for dataset in DATASETS
            ],
        ).sample(20_000, random_state=SEED).reset_index(drop=True)
        df["SMILES_LEN"] = df["CANONICAL_SMILES"].map(len)
        df = df.sort_values("SMILES_LEN")

        smallest_5_percent = df[: math.floor(0.05 * df.shape[0])]
        smallest_5_percent = pd.concat([smallest_5_percent] * 5)

        largest_5_percent = df[math.ceil(0.95 * df.shape[0]) :]
        largest_5_percent = pd.concat([largest_5_percent] * 5)

        Path("data/preprocessed/time_benchmark").mkdir(parents=True, exist_ok=True)
        df.to_csv(
            f"data/preprocessed/time_benchmark/representative-20000.{featurization}.csv",
            index=False,
        )
        df.sample(10_000, random_state=SEED).to_csv(
            f"data/preprocessed/time_benchmark/representative-10000.{featurization}.csv",
            index=False,
        )
        df.sample(5_000, random_state=SEED).to_csv(
            f"data/preprocessed/time_benchmark/representative-5000.{featurization}.csv",
            index=False,
        )
        df.sample(2_500, random_state=SEED).to_csv(
            f"data/preprocessed/time_benchmark/representative-2500.{featurization}.csv",
            index=False,
        )
        smallest_5_percent.to_csv(
            f"data/preprocessed/time_benchmark/small_compounds.{featurization}.csv",
            index=False,
        )
        largest_5_percent.to_csv(
            f"data/preprocessed/time_benchmark/large_compounds.{featurization}.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
