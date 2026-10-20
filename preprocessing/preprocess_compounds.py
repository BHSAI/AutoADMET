import argparse
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem

from trpv1_utils import (
    fingerprints,
    mol_processing,
    scaffold_utils,
    deduplication,
    descriptors,
)
from trpv1_utils.config import COL_SMILES, COL_CLASS, COL_CANONICAL_SMILES, COL_INCHIKEY

SMILES: str = "SMILES"


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Canonicalize smiles
    df[[COL_CANONICAL_SMILES, COL_INCHIKEY]] = df[COL_SMILES].apply(
        lambda smiles: pd.Series(mol_processing.process_smiles(smiles))
    )
    return df.dropna(subset=COL_CANONICAL_SMILES)


def add_morgan_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    # Generate fingerprints
    fps = df[COL_CANONICAL_SMILES].apply(
        lambda smiles: pd.Series(
            (
                smiles,
                *fingerprints.generate_morgan_fp(Chem.MolFromSmiles(smiles)),
            )
        )
    )
    fps.columns = [
        COL_CANONICAL_SMILES,
        *[f"FEATURE_{col}" for col in fps.columns[1:]],
    ]

    # Combine with original df
    return df.merge(right=fps, on=COL_CANONICAL_SMILES)


def add_mordred_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    desc_df = descriptors.compute_mordred_descriptors(
        df[COL_CANONICAL_SMILES].apply(Chem.MolFromSmiles)
    )
    desc_df = descriptors.clean_descriptors(desc_df)
    desc_df.columns = [f"FEATURE_{col}" for col in desc_df.columns]

    return pd.concat([df, desc_df], axis=1)


MORGAN_FP_KEY: str = "morgan_fp"
MORDRED_DESC_KEY: str = "mordred_desc"
featurizers = {
    MORGAN_FP_KEY: add_morgan_fingerprints,
    MORDRED_DESC_KEY: add_mordred_descriptors,
}


def clean_compound_data(
    df: pd.DataFrame,
    smiles_col: str = COL_SMILES,
    class_col: str = COL_CLASS,
    training_data: bool = True,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()

    # Standardize column names
    df.rename({smiles_col: COL_SMILES}, axis=1, inplace=True)

    if training_data:
        df.rename({class_col: COL_CLASS}, axis=1, inplace=True)

    # Canonicalize SMILES
    df = canonicalize(df)

    # Drop duplicates on class column
    if training_data:
        dedup_df: pd.DataFrame
        dedup_df, stats = deduplication.deduplicate_full_pipeline(df)
        print(f"Removed duplicates: {stats}")
        print(f"Size before: {df.shape[0]}. Size after: {dedup_df.shape[0]}")
        df = dedup_df

    cleaning_time = time.perf_counter() - start

    return df, cleaning_time


def split_data(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    # Perform scaffold split
    train_idx, test_idx = scaffold_utils.scaffold_split(df[COL_CANONICAL_SMILES])
    train_compounds = df.iloc[train_idx]
    test_compounds = df.iloc[test_idx]

    # Validate split
    scaffold_utils.validate_split(
        train_compounds,
        test_compounds,
        label_col=COL_CLASS,
    )

    return train_idx, test_idx


def generate_descriptors(
    df: pd.DataFrame,
    featurization: str,
) -> tuple[pd.DataFrame, float]:
    # Generate fingerprints
    start = time.perf_counter()
    df = featurizers[featurization](df)
    featurization_time = time.perf_counter() - start

    return df, featurization_time


def preprocess_compounds(
    raw_data_csv: str,
    output_dir: str,
    training_data: bool = True,
    smiles_col: str = COL_SMILES,
    class_col: str = COL_CLASS,
    featurizations: list[str] = [MORGAN_FP_KEY],
) -> tuple[dict[str, pd.DataFrame], float, dict[str, float]]:
    df = pd.read_csv(raw_data_csv)

    df, cleaning_time = clean_compound_data(df, smiles_col, class_col, training_data)

    # Save canonicalized data
    df[["Name", "SMILES", "CANONICAL_SMILES", "CLASS"]].to_csv(
        f"{output_dir}/canonical_smiles.all.csv", index=False
    )
    if training_data:
        # Split data and save canonicalized splits
        train_idx, test_idx = split_data(df)
        df.iloc[train_idx][["Name", "SMILES", "CANONICAL_SMILES", "CLASS"]].to_csv(
            f"{output_dir}/canonical_smiles.train.csv", index=False
        )
        df.iloc[test_idx][["Name", "SMILES", "CANONICAL_SMILES", "CLASS"]].to_csv(
            f"{output_dir}/canonical_smiles.test.csv", index=False
        )

    featurized_dfs = {}
    featurization_times = {}

    for featurization in featurizations:
        # Generate fingerprints
        start = time.perf_counter()
        featurized_df = featurizers[featurization](df)
        featurization_time = time.perf_counter() - start

        # Save featurized data
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        featurized_df.to_csv(f"{output_dir}/{featurization}.all.csv", index=False)

        # Perform and save scaffold split
        if training_data:
            # Save split dataframes
            train_compounds = featurized_df.iloc[train_idx]
            test_compounds = featurized_df.iloc[test_idx]
            train_compounds.to_csv(
                f"{output_dir}/{featurization}.train.csv",
                index=False,
            )
            test_compounds.to_csv(
                f"{output_dir}/{featurization}.test.csv",
                index=False,
            )

        featurized_dfs[featurization] = featurized_df
        featurization_times[featurization] = featurization_time

    return featurized_dfs, cleaning_time, featurization_times


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preprocess_compounds.py",
        description="Preprocess and featurize compound data",
    )
    parser.add_argument(
        "--raw_data",
        "-d",
        help="The raw compound data",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="The directory to save the featurized compound data to",
    )
    parser.add_argument(
        "--query_data",
        "-q",
        action="store_true",
        help="Whether this data is training data",
    )
    parser.add_argument(
        "--smiles_col",
        "-s",
        default=COL_SMILES,
        help="Name of the column containing SMILES",
    )
    parser.add_argument(
        "--class_col",
        "-c",
        default=COL_CLASS,
        help="Name of the classification column for training data",
    )
    parser.add_argument(
        "--featurization",
        "-f",
        type=str,
        choices=[MORGAN_FP_KEY, MORDRED_DESC_KEY],
        default=MORGAN_FP_KEY,
        help="Which featurization to use",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    preprocess_compounds(
        raw_data_csv=args.raw_data,
        output_dir=args.output_dir,
        training_data=not args.query_data,
        smiles_col=args.smiles_col,
        class_col=args.class_col,
        featurizations=[args.featurization],
    )


if __name__ == "__main__":
    main()
