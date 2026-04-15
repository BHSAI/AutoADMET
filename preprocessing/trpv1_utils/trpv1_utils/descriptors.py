"""
Molecular descriptor utilities for TRPV1 ML benchmark.

Provides functions for calculating and preprocessing Mordred molecular descriptors.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

def check_mordred_available():
    """Check if mordred is available."""
    try:
        from mordred import Calculator, descriptors
        return True
    except ImportError:
        return False

def compute_mordred_descriptors(mols):
    """
    Compute Mordred descriptors for RDKit molecule objects.

    Args:
        mols: List of RDKit Mol objects

    Returns:
        DataFrame with descriptor values (rows=molecules, cols=descriptors)
    """
    if not check_mordred_available():
        raise ImportError("mordred package not installed. Install with: pip install mordred")

    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)

    try:
        mol_series = pd.Series(mols)
        desc_df = calc.pandas(mol_series)
        return desc_df
    except Exception as e:
        warnings.warn(f"Mordred descriptor calculation failed: {e}")
        return pd.DataFrame(index=range(len(mols)))

def clean_descriptors(desc_df, variance_threshold=0.01, correlation_threshold=0.95,
                     handle_missing='drop', verbose=True):
    """
    Clean and preprocess Mordred descriptors.

    Steps:
    1. Handle missing/infinite values
    2. Remove constant features (low variance)
    3. Remove highly correlated features

    Args:
        desc_df: DataFrame with raw descriptors
        variance_threshold: Minimum variance for feature retention
        correlation_threshold: Max correlation for feature retention
        handle_missing: 'drop' columns with missing, 'fill' with median, or 'none'
        verbose: Print cleaning statistics

    Returns:
        Cleaned descriptor DataFrame
    """
    if desc_df.empty:
        return desc_df

    original_shape = desc_df.shape

    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    if handle_missing == 'drop':
        desc_df = desc_df.dropna(axis=1)
    elif handle_missing == 'fill':
        desc_df = desc_df.fillna(desc_df.median())

    desc_df = desc_df.select_dtypes(include=[np.number])

    variances = desc_df.var()
    keep_cols = variances[variances > variance_threshold].index
    desc_df = desc_df[keep_cols]

    if correlation_threshold < 1.0:
        corr_matrix = desc_df.corr(method="spearman").abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [col for col in upper_tri.columns
                   if (upper_tri[col] > correlation_threshold).any()]

        desc_df = desc_df.drop(columns=to_drop)

    if verbose:
        print(f"Descriptor cleaning:")
        print(f"  Original: {original_shape[0]} samples × {original_shape[1]} descriptors")
        print(f"  After cleaning: {desc_df.shape[0]} samples × {desc_df.shape[1]} descriptors")
        if handle_missing == 'drop':
            print(f"  Removed {original_shape[1] - desc_df.shape[1]} features")

    return desc_df

def save_descriptor_info(desc_df, scaler, output_dir, endpoint, prefix="Mordred"):
    """
    Save descriptor feature list and fitted scaler for later use.

    Args:
        desc_df: Cleaned descriptor DataFrame
        scaler: Fitted sklearn scaler object
        output_dir: Directory to save files
        endpoint: IC50 or EC50
        prefix: Descriptor type prefix
    """
    import pickle

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_file = output_dir / f"{endpoint}_{prefix}_features.pkl"
    with open(features_file, 'wb') as f:
        pickle.dump(list(desc_df.columns), f)

    scaler_file = output_dir / f"{endpoint}_{prefix}_scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Saved feature list to: {features_file}")
    print(f"Saved scaler to: {scaler_file}")

def load_descriptor_info(output_dir, endpoint, prefix="Mordred"):
    """
    Load saved descriptor feature list and scaler.

    Args:
        output_dir: Directory with saved files
        endpoint: IC50 or EC50
        prefix: Descriptor type prefix

    Returns:
        Tuple of (feature_list, scaler)
    """
    import pickle

    output_dir = Path(output_dir)

    features_file = output_dir / f"{endpoint}_{prefix}_features.pkl"
    scaler_file = output_dir / f"{endpoint}_{prefix}_scaler.pkl"

    with open(features_file, 'rb') as f:
        features = pickle.load(f)

    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    return features, scaler
