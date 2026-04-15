"""
Shared utilities for TRPV1 ML benchmark preprocessing and analysis.

This package contains reusable functions for:
- Molecular validation and standardization (mol_processing)
- Duplicate detection and removal (deduplication)
- Scaffold-based dataset splitting (scaffold_utils)
- Fingerprint generation (fingerprints)
- ML model factories (models)
- Evaluation metrics (evaluation)
- Configuration and path management (config)
"""

from .config import (
    REPO_ROOT,
    DATA_RAW,
    DATA_INTERMEDIATE,
    DATA_PREPROCESSED,
    RESULTS_ROOT,
    MODELS_ROOT,
    FIGURES_ROOT,
    ELEMENT_WHITELIST,
    MORGAN_RADIUS,
    MORGAN_FP_SIZE,
    SPLIT_SIZES,
    RANDOM_SEED,
    BASE_RANDOM_STATE,
    N_REPEATS,
    N_FOLDS,
    FINGERPRINT_TYPES,
    MODEL_ORDER,
    COL_SMILES,
    COL_CANONICAL_SMILES,
    COL_INCHIKEY,
    COL_CLASS,
    COL_ID,
    get_raw_file,
    get_standardized_file,
    get_deduplicated_file,
    get_train_file,
    get_test_file,
    get_results_dir,
    get_model_path,
    get_figure_path,
)

__all__ = [
    "REPO_ROOT",
    "DATA_RAW",
    "DATA_INTERMEDIATE",
    "DATA_PREPROCESSED",
    "RESULTS_ROOT",
    "MODELS_ROOT",
    "FIGURES_ROOT",
    "ELEMENT_WHITELIST",
    "MORGAN_RADIUS",
    "MORGAN_FP_SIZE",
    "SPLIT_SIZES",
    "RANDOM_SEED",
    "BASE_RANDOM_STATE",
    "N_REPEATS",
    "N_FOLDS",
    "FINGERPRINT_TYPES",
    "MODEL_ORDER",
    "COL_SMILES",
    "COL_CANONICAL_SMILES",
    "COL_INCHIKEY",
    "COL_CLASS",
    "COL_ID",
    "get_raw_file",
    "get_standardized_file",
    "get_deduplicated_file",
    "get_train_file",
    "get_test_file",
    "get_results_dir",
    "get_model_path",
    "get_figure_path",
]
