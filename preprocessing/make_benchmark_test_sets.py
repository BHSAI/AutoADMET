from pathlib import Path

import pandas as pd
import math

FEATURIZATIONS = ["morgan_fp", "mordred_desc"]


def main():
    for featurization in FEATURIZATIONS:
        df = pd.read_csv(f"data/preprocessed/cyp3a4/{featurization}.all.csv")
        df["SMILES_LEN"] = df["CANONICAL_SMILES"].map(len)
        df = df.sort_values("SMILES_LEN")

        smallest_5_percent = df[: math.floor(0.05 * df.shape[0])]
        largest_5_percent = df[math.ceil(0.95 * df.shape[0]) :]

        Path("data/preprocessed/time_benchmark").mkdir(parents=True, exist_ok=True)
        smallest_5_percent.to_csv(
            f"data/preprocessed/time_benchmark/small_compounds.{featurization}.csv"
        )
        largest_5_percent.to_csv(
            f"data/preprocessed/time_benchmark/large_compounds.{featurization}.csv"
        )


if __name__ == "__main__":
    main()
