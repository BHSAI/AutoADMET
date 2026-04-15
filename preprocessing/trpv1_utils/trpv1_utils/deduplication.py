"""
Duplicate detection and removal utilities.

This module provides functions for:
- Detecting duplicates by InChIKey (connectivity layer)
- Detecting duplicates by stereo-agnostic tautomer comparison
- Majority-vote based duplicate resolution (for conflicting labels)
- Complete deduplication pipeline
"""

import pandas as pd
from collections import Counter
from .mol_processing import extract_inchikey14, nostereo_tautomer
from .config import COL_INCHIKEY, COL_INCHIKEY14, COL_CANONICAL_SMILES, COL_CLASS, COL_TAUT_NONISO


# ============================================================================
# Core Deduplication Functions
# ============================================================================

def drop_duplicates_by_majority(df, key_col, label_col):
    """
    Remove duplicates using majority vote on labels.

    When multiple entries have the same key (e.g., InChIKey14) but different
    labels (e.g., active/inactive), keep only one entry with the majority label.
    If there's a tie, keep the first occurrence.

    Args:
        df: DataFrame to deduplicate
        key_col: Column name to use for duplicate detection (e.g., 'InChIKey14')
        label_col: Column name containing labels for majority vote (e.g., 'CLASS')

    Returns:
        Tuple of (deduplicated_df, n_dropped)
    """
    # Find all duplicated entries based on key_col
    dup_groups = df[df.duplicated(key_col, keep=False)].groupby(key_col)

    to_drop = []

    for key, group in dup_groups:
        # Count occurrences of each label
        label_counts = Counter(group[label_col])

        # Get most common label (if tie, first one is chosen)
        mode_label, freq = label_counts.most_common(1)[0]

        # Keep first occurrence with the majority label
        keep_idx = group[group[label_col] == mode_label].index[0]

        # Mark all others for removal
        drop_idx = [i for i in group.index if i != keep_idx]
        to_drop.extend(drop_idx)

    # Drop duplicates and return
    df_dedup = df.drop(index=to_drop)
    n_dropped = len(to_drop)

    return df_dedup, n_dropped


# ============================================================================
# InChIKey-based Deduplication
# ============================================================================

def deduplicate_by_inchikey14(df, label_col=COL_CLASS):
    """
    Remove stereo/isotope duplicates using InChIKey connectivity layer.

    The first 14 characters of InChIKey represent molecular connectivity
    without stereochemistry or isotope information. This function removes
    duplicates that differ only in stereo/isotope.

    Args:
        df: DataFrame with InChIKey column
        label_col: Column name for majority vote (default: 'CLASS')

    Returns:
        Tuple of (deduplicated_df, n_dropped)
    """
    # Add InChIKey14 column (connectivity layer only)
    df[COL_INCHIKEY14] = df[COL_INCHIKEY].map(extract_inchikey14)

    # Deduplicate using majority vote
    df_dedup, n_dropped = drop_duplicates_by_majority(df, COL_INCHIKEY14, label_col)

    # Remove temporary column
    df_dedup = df_dedup.drop(columns=[COL_INCHIKEY14])

    return df_dedup, n_dropped


# ============================================================================
# Tautomer-based Deduplication
# ============================================================================

def deduplicate_by_tautomer(df, label_col=COL_CLASS):
    """
    Remove tautomer duplicates (stereo-agnostic).

    This function detects duplicates that are different tautomers of the same
    molecule, ignoring stereochemistry.

    Args:
        df: DataFrame with CANONICAL_SMILES column
        label_col: Column name for majority vote (default: 'CLASS')

    Returns:
        Tuple of (deduplicated_df, n_dropped)
    """
    # Add tautomer non-isomeric column (no stereo, canonical tautomer)
    df[COL_TAUT_NONISO] = df[COL_CANONICAL_SMILES].map(nostereo_tautomer)

    # Deduplicate using majority vote
    df_dedup, n_dropped = drop_duplicates_by_majority(df, COL_TAUT_NONISO, label_col)

    # Remove temporary column
    df_dedup = df_dedup.drop(columns=[COL_TAUT_NONISO])

    return df_dedup, n_dropped


# ============================================================================
# Complete Deduplication Pipeline
# ============================================================================

def deduplicate_full_pipeline(df, label_col=COL_CLASS):
    """
    Complete deduplication pipeline.

    Steps:
    1. Remove stereo/isotope duplicates (InChIKey14)
    2. Remove tautomer duplicates (stereo-agnostic)
    3. Reset index

    Args:
        df: DataFrame to deduplicate
        label_col: Column name for majority vote (default: 'CLASS')

    Returns:
        Deduplicated DataFrame with reset index
    """
    # Step 1: Remove stereo/isotope duplicates
    df, n_stereo = deduplicate_by_inchikey14(df, label_col)

    # Step 2: Remove tautomer duplicates
    df, n_taut = deduplicate_by_tautomer(df, label_col)

    # Reset index
    df = df.reset_index(drop=True)

    # Log summary
    total_removed = n_stereo + n_taut

    return df, {
        "stereo_isotope_duplicates": n_stereo,
        "tautomer_duplicates": n_taut,
        "total_removed": total_removed
    }
