import pandas as pd
from rdkit import Chem
from pathlib import Path
import argparse

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


def preprocess_compounds(
    raw_data_csv: str,
    output_dir: str,
    training_data: bool = True,
    smiles_col: str = COL_SMILES,
    class_col: str = COL_CLASS,
    featurization: str = MORGAN_FP_KEY,
) -> None:
    df = pd.read_csv(raw_data_csv)

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

    # Generate fingerprints
    df = featurizers[featurization](df)

    # Save featurized data
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    df.to_csv(f"{output_dir}/{featurization}.all.csv", index=False)

    # Perform and save scaffold split
    if training_data:
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

        # Save split dataframes
        train_compounds.to_csv(f"{output_dir}/{featurization}.train.csv", index=False)
        test_compounds.to_csv(f"{output_dir}/{featurization}.test.csv", index=False)


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
        featurization=args.featurization,
    )


if __name__ == "__main__":
    main()
