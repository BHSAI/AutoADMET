"""
Shared configuration for TRPV1 ML benchmark preprocessing and analysis.

This module defines:
- Repository paths (data directories, results, models, figures)
- Molecular processing parameters (element whitelist, fingerprint settings)
- Data splitting parameters (train/test ratios, random seed)
- Standard column names used across the pipeline
"""

from pathlib import Path

# ============================================================================
# Repository Paths
# ============================================================================

# Repository root - works from any script location
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_ROOT = REPO_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_INTERMEDIATE = DATA_ROOT / "intermediate"
DATA_PREPROCESSED = DATA_ROOT / "pre-processed"

# Output directories
RESULTS_ROOT = REPO_ROOT / "results"
MODELS_ROOT = REPO_ROOT / "models"
FIGURES_ROOT = REPO_ROOT / "figures"

# Ensure critical directories exist
DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
DATA_PREPROCESSED.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Molecular Processing Parameters
# ============================================================================

# Elements allowed in organic molecules (non-metals + common halogens)
ELEMENT_WHITELIST = {"H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

# Morgan fingerprint parameters
MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048
MORGAN_USE_BOND_TYPES = True
MORGAN_INCLUDE_CHIRALITY = False
MORGAN_COUNT_SIMULATION = False

# ============================================================================
# Data Splitting Parameters
# ============================================================================

# Train/test split ratio (80/20)
SPLIT_SIZES = (0.8, 0.2)

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Standard Column Names
# ============================================================================

# Input/output column names used across preprocessing pipeline
COL_SMILES = "SMILES"
COL_CANONICAL_SMILES = "CANONICAL_SMILES"
COL_INCHIKEY = "InChIKey"
COL_INCHIKEY14 = "InChIKey14"
COL_CLASS = "CLASS"
COL_ID = "ID"
COL_TAUT_NONISO = "TAUT_NONISO"

# ============================================================================
# File Naming Conventions
# ============================================================================

def get_raw_file(endpoint: str) -> Path:
    """Get path to the cleaned input dataset used as preprocessing entrypoint.

    Note: In this repo layout, the "raw" ChEMBL exports are stored as XLSX under
    data/raw/*_raw_data/, while the cleaned CSV inputs are stored under
    data/intermediate/*_intermediate/. The preprocessing pipeline starts from
    those cleaned CSVs.

    Args:
        endpoint: 'IC50' or 'EC50'

    Returns:
        Path to cleaned input CSV file
    """
    endpoint = endpoint.upper()
    if endpoint not in {"IC50", "EC50"}:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    cleaned_dir = DATA_INTERMEDIATE / f"{endpoint}_intermediate"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    return cleaned_dir / f"TRPV1_chembl_{endpoint}_cleaned_v1.csv"

def get_standardized_file(endpoint: str) -> Path:
    """Get path to standardized (RDKit cleaned) file.

    Args:
        endpoint: 'IC50' or 'EC50'

    Returns:
        Path to standardized CSV file
    """
    return DATA_INTERMEDIATE / f"TRPV1_{endpoint}_cleaned_RDK.csv"


def get_deduplicated_file(endpoint: str) -> Path:
    """Get path to deduplicated file (optional step).

    Args:
        endpoint: 'IC50' or 'EC50'

    Returns:
        Path to deduplicated CSV file
    """
    return DATA_INTERMEDIATE / f"TRPV1_{endpoint}_cleaned_RDK_dedup.csv"


def get_train_file(endpoint: str) -> Path:
    """Get path to training set after scaffold split.

    Args:
        endpoint: 'IC50' or 'EC50'

    Returns:
        Path to training CSV file
    """
    return DATA_PREPROCESSED / f"TRPV1_{endpoint}_train_scaffold.csv"


def get_test_file(endpoint: str) -> Path:
    """Get path to external test set after scaffold split.

    Args:
        endpoint: 'IC50' or 'EC50'

    Returns:
        Path to external test CSV file
    """
    return DATA_PREPROCESSED / f"TRPV1_{endpoint}_exttest_scaffold.csv"


def get_similarity_pairs_file(endpoint: str) -> Path:
    """Get path to similarity analysis output (QC only).

    Args:
        endpoint: 'IC50' or 'EC50'

    Returns:
        Path to identical pairs CSV file
    """
    return DATA_INTERMEDIATE / f"TRPV1_{endpoint}_identical_pairs.csv"


# ============================================================================
# Analysis Configuration
# ============================================================================

# Cross-validation parameters
N_REPEATS = 5
N_FOLDS = 5
BASE_RANDOM_STATE = 42

# Fingerprint types for analysis
FINGERPRINT_TYPES = ["RDKITfp", "MACCS", "Morgan", "Mordred"]

# Model execution order for consistent reporting
MODEL_ORDER = [
    "KNN", "SVM", "Bayesian", "LogisticRegression",
    "RandomForest", "LightGBM", "XGBoost",
]

# ============================================================================
# Analysis Output Paths
# ============================================================================

def get_results_dir(endpoint: str, fingerprint_type: str = None) -> Path:
    """Get results directory for endpoint and fingerprint type.

    Args:
        endpoint: 'IC50' or 'EC50'
        fingerprint_type: Optional fingerprint type ('RDKITfp', 'Morgan', 'MACCS', 'Mordred')

    Returns:
        Path to results directory
    """
    if fingerprint_type:
        result_dir = RESULTS_ROOT / f"{endpoint}_results" / f"TRPV1_{endpoint}_{fingerprint_type}"
    else:
        result_dir = RESULTS_ROOT / f"{endpoint}_results"

    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def get_model_path(endpoint: str, fingerprint_type: str, model_name: str) -> Path:
    """Get path for saved model file.

    Args:
        endpoint: 'IC50' or 'EC50'
        fingerprint_type: Fingerprint type used
        model_name: Model name (e.g., 'LogisticRegression')

    Returns:
        Path to model file (.pkl)
    """
    return MODELS_ROOT / f"TRPV1_{endpoint}_{fingerprint_type}_{model_name}_final_model.pkl"


def get_figure_path(endpoint: str, figure_name: str) -> Path:
    """Get path for saved figure file.

    Args:
        endpoint: 'IC50' or 'EC50'
        figure_name: Name of the figure

    Returns:
        Path to figure file
    """
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    return FIGURES_ROOT / f"{endpoint}_{figure_name}"
