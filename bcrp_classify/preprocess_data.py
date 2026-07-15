#!/usr/bin/env python
"""Standardize and curate SMILES, perform Butina cluster-based train/test split, and plot similarity distributions."""

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

from config import Config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
import seaborn as sns

import logging

logger = logging.getLogger(__name__)

def _import_preprocessing():
    """Lazy import — avoids chembl_structure_pipeline when only plotting."""
    from utils.preprocessing import (
        preprocess_dataset,
        calculate_train_test_similarity,
        get_preprocessing_statistics,
    )
    return preprocess_dataset, calculate_train_test_similarity, get_preprocessing_statistics

warnings.filterwarnings('ignore')


# ============================================================================
# Visualization
# ============================================================================

def plot_similarity_distribution(similarities: np.ndarray, 
                                  output_file: Path,
                                  title: str = "Train-Test Tanimoto Similarity"):
    """
    Create histogram of Tanimoto similarities.
    
    Args:
        similarities: Array of similarity values
        output_file: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    ax = sns.histplot(similarities, bins=40, color='red', kde=True)
    ax.set_xlabel("Tanimoto Similarity (Morgan FPs)", fontsize=18)
    ax.set_ylabel("Count", fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(labelsize=14)

    # Add statistics
    mean_sim = similarities.mean()
    median_sim = np.median(similarities)
    ax.axvline(mean_sim, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_sim:.3f}')
    ax.axvline(median_sim, color='green', linestyle='--', linewidth=2,
               label=f'Median: {median_sim:.3f}')
    ax.legend(frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved similarity plot: {output_file.name}")


def plot_umap_projection(train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         output_file: Path,
                         smiles_col: str = 'SMILES',
                         radius: int = 2,
                         nbits: int = 2048,
                         seed: int = 42):
    """
    UMAP projection of Morgan fingerprints coloured by train/test membership.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        output_file: Path to save figure
        smiles_col: Name of SMILES column
        radius: Morgan radius
        nbits: Morgan bit length
        seed: Random seed for UMAP
    """
    try:
        import umap as umap_lib
    except ImportError:
        logger.info("  Skipping UMAP plot (umap-learn not installed)")
        return

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    def to_fps(smiles_list):
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(gen.GetFingerprintAsNumPy(mol))
        return np.array(fps)

    X_train = to_fps(train_df[smiles_col].tolist())
    X_test  = to_fps(test_df[smiles_col].tolist())

    reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.3, metric="jaccard", random_state=seed)
    train_emb = reducer.fit_transform(X_train)
    test_emb  = reducer.transform(X_test)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(train_emb[:, 0], train_emb[:, 1],
               s=12, alpha=0.35, color="red", label=f"Train (n={len(X_train)})",
               edgecolors="none", rasterized=True)
    ax.scatter(test_emb[:, 0], test_emb[:, 1],
               s=22, alpha=0.7, color="green", label=f"Test (n={len(X_test)})",
               edgecolors="none", rasterized=True)
    ax.set_xlabel("UMAP-1", fontsize=20)
    ax.set_ylabel("UMAP-2", fontsize=20)
    ax.legend(frameon=False, fontsize=14)
    ax.tick_params(labelsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved UMAP projection: {output_file.name}")


def plot_class_distribution(train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             output_file: Path,
                             label_col: str = 'CLASS'):
    """
    Create grouped bar plot comparing class distributions in train and test.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        output_file: Path to save figure
        label_col: Name of label column
    """
    train_counts = train_df[label_col].value_counts().sort_index()
    test_counts = test_df[label_col].value_counts().sort_index()

    sets = ['Train', 'Test']
    non_inh = [train_counts.get(0, 0), test_counts.get(0, 0)]
    inh = [train_counts.get(1, 0), test_counts.get(1, 0)]

    x = np.arange(len(sets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars_non = ax.bar(x - width / 2, non_inh, width,
                      label='Non-inhibitors', color='steelblue')
    bars_inh = ax.bar(x + width / 2, inh, width,
                      label='Inhibitors', color='coral')

    # Annotate counts on bars
    for bars in [bars_non, bars_inh]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    str(int(h)), ha='center', va='bottom', fontsize=14)

    ax.set_ylabel('Count', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(sets)
    ax.tick_params(labelsize=14)
    ax.legend(frameon=False, fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved class distribution: {output_file.name}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess molecular toxicity data for ML pipeline",
    )

    parser.add_argument('--input', '-i', default=None,
                        help='Input CSV file with SMILES and CLASS columns')
    parser.add_argument('--output-dir', '-o', default='preprocessed',
                        help='Output directory for processed files (default: preprocessed)')

    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--smiles-col', default='SMILES',
                        help='Name of SMILES column (default: SMILES)')
    parser.add_argument('--label-col', default='CLASS',
                        help='Name of label column (default: CLASS)')

    parser.add_argument('--no-validate', action='store_true',
                        help='Skip structure validation')
    parser.add_argument('--no-standardize', action='store_true',
                        help='Skip SMILES standardization')
    parser.add_argument('--no-butina', action='store_true',
                        help='Skip Butina clustering (use random split)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--plot-only', nargs=2, metavar=('TRAIN_CSV', 'TEST_CSV'),
                        help='Skip preprocessing; just generate similarity and class '
                             'distribution plots from existing train/test CSVs')

    args = parser.parse_args()

    start_time = datetime.now()

    if args.plot_only:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import BulkTanimotoSimilarity

        train_csv, test_csv = args.plot_only
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nPLOT-ONLY MODE")
        logger.info(f"  Train: {train_csv} ({len(train_df)} molecules)")
        logger.info(f"  Test:  {test_csv} ({len(test_df)} molecules)")

        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=Config.MORGAN_RADIUS, fpSize=Config.MORGAN_NBITS)
        train_fps = [fp_gen.GetFingerprint(Chem.MolFromSmiles(s))
                     for s in train_df[args.smiles_col] if Chem.MolFromSmiles(s)]
        top_n = 5
        sims = []
        for smi in test_df[args.smiles_col]:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = fp_gen.GetFingerprint(mol)
            ts = BulkTanimotoSimilarity(fp, train_fps)
            sims.extend(sorted(ts, reverse=True)[:top_n])
        similarities = np.array(sims)

        plot_similarity_distribution(similarities, output_dir / 'similarity_distribution.png')
        plot_class_distribution(train_df, test_df, output_dir / 'class_distribution.png',
                                args.label_col)
        plot_umap_projection(train_df, test_df, output_dir / 'umap_train_test.png',
                             smiles_col=args.smiles_col)
        logger.info("Done.")
        return
    
    if args.input is None:
        parser.error("--input is required when not using --plot-only")

    logger.info("\nMOLECULAR DATA PREPROCESSING")
    logger.info(f"Started: {start_time:%Y-%m-%d %H:%M:%S}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        logger.info(f"\nLoading data from {args.input}...")
        df = pd.read_csv(args.input)
        
        # Validate columns
        if args.smiles_col not in df.columns:
            raise ValueError(f"Column '{args.smiles_col}' not found. Available: {df.columns.tolist()}")
        if args.label_col not in df.columns:
            raise ValueError(f"Column '{args.label_col}' not found. Available: {df.columns.tolist()}")
        
        logger.info(f"  Loaded {len(df)} molecules")
        logger.info(f"  Columns: {df.columns.tolist()}")
        
        # Run preprocessing
        preprocess_dataset, calculate_train_test_similarity, get_preprocessing_statistics = _import_preprocessing()
        train_df, test_df, stats = preprocess_dataset(
            df,
            test_size=args.test_size,
            smiles_col=args.smiles_col,
            label_col=args.label_col,
            validate=not args.no_validate,
            standardize=not args.no_standardize,
            use_butina=not args.no_butina,
            random_state=args.random_state
        )
        
        train_file = output_dir / 'train_df.csv'
        test_file = output_dir / 'test_df.csv'
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        logger.info(f"\nSaved: {train_file} ({len(train_df)} molecules)")
        logger.info(f"Saved: {test_file} ({len(test_df)} molecules)")

        stats_json = output_dir / 'preprocessing_stats.json'
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        stats_df = get_preprocessing_statistics(stats)
        stats_csv = output_dir / 'preprocessing_stats.csv'
        stats_df.to_csv(stats_csv)

        if not args.no_plots:
            similarities = calculate_train_test_similarity(
                train_df[args.smiles_col].tolist(),
                test_df[args.smiles_col].tolist(),
                top_n=5
            )
            plot_similarity_distribution(similarities, output_dir / 'similarity_distribution.png')
            plot_class_distribution(train_df, test_df, output_dir / 'class_distribution.png',
                                    args.label_col)
            plot_umap_projection(train_df, test_df, output_dir / 'umap_train_test.png',
                                 smiles_col=args.smiles_col)

        end_time = datetime.now()
        logger.info(f"\nPREPROCESSING COMPLETED  ({end_time - start_time})")
        logger.info(f"Output: {args.output_dir}")
        
    except Exception as e:
        logger.info(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
