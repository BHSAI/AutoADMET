"""Molecular standardization, SMILES validation, and Butina-cluster-based train/test splitting."""

import logging
import warnings
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.DataStructs import BulkTanimotoSimilarity

from chembl_structure_pipeline import standardizer, exclude_flag
import useful_rdkit_utils as uru

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Module-level SaltRemover (avoid re-instantiation per molecule)
_SALT_REMOVER = SaltRemover()


# ============================================================================
# Structure Validation and Standardization
# ============================================================================

def is_valid_structure(smiles: str) -> bool:
    """
    Check if a SMILES string represents a valid, non-excluded chemical structure.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
        
    Notes:
        Uses ChEMBL exclude_flag to filter problematic structures
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        exclude = exclude_flag.exclude_flag(mol)
        return not exclude
            
    except Exception as e:
        logger.debug(f"Validation error for {smiles}: {e}")
        return False


def standardize_and_canonicalize(smiles: str) -> Optional[str]:
    """
    Standardize and canonicalize a SMILES string using ChEMBL pipeline.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Standardized canonical SMILES, or None if standardization fails
        
    Process:
        1. Remove salts using ChEMBL get_parent_mol
        2. Remove remaining salts with RDKit SaltRemover
        3. Standardize using ChEMBL standardize_mol
        4. Generate canonical SMILES
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        
        m_no_salts = standardizer.get_parent_mol(molecule)
        to_standardize = m_no_salts[0]
        stripped = _SALT_REMOVER.StripMol(to_standardize)
        std_mol = standardizer.standardize_mol(stripped)
        canonical_smiles = Chem.MolToSmiles(std_mol)
        
        return canonical_smiles
        
    except Exception as e:
        logger.debug(f"Standardization error for {smiles}: {e}")
        return None


# ============================================================================
# Duplicate Removal
# ============================================================================

def remove_duplicates(df: pd.DataFrame, 
                      smiles_col: str = 'SMILES',
                      label_col: str = 'CLASS') -> pd.DataFrame:
    """
    Remove duplicate molecules from dataset.
    
    Args:
        df: Input dataframe
        smiles_col: Name of SMILES column
        label_col: Name of label column
        
    Returns:
        DataFrame with duplicates removed
        
    Process:
        1. Remove duplicates with same SMILES and same label
        2. Remove molecules with conflicting labels (keep=False)
    """
    df_no_dup = df.drop_duplicates(subset=[smiles_col, label_col])
    logger.info(f"  After removing same-label duplicates: {len(df_no_dup)}")
    df_final = df_no_dup.drop_duplicates(subset=[smiles_col], keep=False)
    logger.info(f"  After removing conflicting labels: {len(df_final)}")
    
    return df_final.reset_index(drop=True)


# ============================================================================
# Butina Clustering
# ============================================================================

def add_butina_clusters(df: pd.DataFrame, 
                        smiles_col: str = 'SMILES',
                        cluster_col: str = 'Butina') -> pd.DataFrame:
    """
    Add Butina cluster assignments to dataframe.
    
    Args:
        df: Input dataframe with SMILES
        smiles_col: Name of SMILES column
        cluster_col: Name for cluster column
        
    Returns:
        DataFrame with added cluster column
        
    Notes:
        Requires useful_rdkit_utils package
        Uses Butina clustering for structural similarity
    """
    logger.info(f"  Running Butina clustering on {len(df)} molecules...")
    df[cluster_col] = uru.get_butina_clusters(df[smiles_col])
    
    n_clusters = df[cluster_col].nunique()
    logger.info(f"  Found {n_clusters} Butina clusters")
    
    return df


# ============================================================================
# Train/Test Splitting by Cluster
# ============================================================================

def split_by_clusters(df: pd.DataFrame,
                      test_size: float = 0.2,
                      cluster_col: str = 'Butina',
                      tolerance: float = 0.05,
                      max_tries: int = 10000,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/test by Butina clusters.
    
    Args:
        df: Input dataframe with cluster assignments
        test_size: Fraction of data for test set (0-1)
        cluster_col: Name of cluster column
        tolerance: Acceptable deviation from target test_size
        max_tries: Maximum attempts to find valid split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
        
    Notes:
        Keeps clusters intact (no cluster split across train/test)
        Iteratively selects clusters to match target test size
    """
    import random
    rng = random.Random(random_state)

    cluster_counts = df[cluster_col].value_counts()
    cluster_sizes = cluster_counts.to_dict()
    all_clusters = list(cluster_sizes.keys())
    total_size = len(df)
    target_test_size = int(total_size * test_size)
    tolerance_abs = int(target_test_size * tolerance)
    
    logger.info(f"\n  Target test size: {target_test_size} ± {tolerance_abs}")
    logger.info(f"  Total clusters: {len(all_clusters)}")
    
    for attempt in range(max_tries):
        rng.shuffle(all_clusters)
        selected_clusters = []
        current_size = 0
        
        for cluster in all_clusters:
            cluster_size = cluster_sizes[cluster]
            
            if current_size + cluster_size <= target_test_size + tolerance_abs:
                selected_clusters.append(cluster)
                current_size += cluster_size
                
                # Check if we're within tolerance
                if abs(current_size - target_test_size) <= tolerance_abs:
                    # Success!
                    mask = df[cluster_col].isin(selected_clusters)
                    test_df = df[mask].reset_index(drop=True)
                    train_df = df[~mask].reset_index(drop=True)
                    
                    logger.info(f"  Found valid split after {attempt + 1} attempts")
                    logger.info(f"  Test size: {len(test_df)} ({100*len(test_df)/total_size:.1f}%)")
                    logger.info(f"  Train size: {len(train_df)} ({100*len(train_df)/total_size:.1f}%)")
                    
                    return train_df, test_df
    
    warnings.warn(
        f"Could not find split within tolerance after {max_tries} attempts. "
        "Using simple cluster assignment."
    )
    
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    selected_clusters = []
    current_size = 0
    
    for cluster, size in sorted_clusters:
        if current_size < target_test_size:
            selected_clusters.append(cluster)
            current_size += size
    
    mask = df[cluster_col].isin(selected_clusters)
    test_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)
    
    return train_df, test_df


# ============================================================================
# Tanimoto Similarity Analysis
# ============================================================================

def calculate_train_test_similarity(train_smiles: List[str],
                                     test_smiles: List[str],
                                     top_n: int = 5) -> np.ndarray:
    """
    Calculate Tanimoto similarities between train and test sets.
    
    Args:
        train_smiles: List of training SMILES
        test_smiles: List of test SMILES
        top_n: Number of most similar training compounds per test compound
        
    Returns:
        Array of top-N similarities for all test compounds
        
    Notes:
        Uses Morgan fingerprints (radius 2, 2048 bits)
        Returns flattened array of all top-N similarities
    """
    logger.info(f"\n  Calculating Tanimoto similarities (top {top_n})...")
    
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
    train_fps = [fp_gen.GetFingerprint(mol) for mol in train_mols if mol is not None]
    
    test_mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
    test_fps = [fp_gen.GetFingerprint(mol) for mol in test_mols if mol is not None]
    
    result_list = []
    for test_fp in tqdm(test_fps, desc="  Tanimoto similarity", leave=False):
        sim_list = BulkTanimotoSimilarity(test_fp, train_fps)
        sim_array = np.array(sim_list)
        
        idx = np.argpartition(sim_array, -top_n)[-top_n:]
        best_n = sim_array[idx]
        result_list.append(best_n)
    
    similarities = np.array(result_list).flatten()
    
    logger.info(f"  Mean similarity: {similarities.mean():.3f}")
    logger.info(f"  Median similarity: {np.median(similarities):.3f}")
    logger.info(f"  Max similarity: {similarities.max():.3f}")
    
    return similarities


# ============================================================================
# Full Preprocessing Pipeline
# ============================================================================

def preprocess_dataset(input_df: pd.DataFrame,
                       test_size: float = 0.2,
                       smiles_col: str = 'SMILES',
                       label_col: str = 'CLASS',
                       validate: bool = True,
                       standardize: bool = True,
                       use_butina: bool = True,
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Complete preprocessing pipeline for molecular datasets.
    
    Args:
        input_df: Raw input dataframe
        test_size: Fraction for test set
        smiles_col: Name of SMILES column
        label_col: Name of label column
        validate: Whether to validate structures
        standardize: Whether to standardize SMILES
        use_butina: Whether to use Butina clustering for split
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df, stats_dict)
        
    Pipeline:
        1. Validate structures
        2. Standardize and canonicalize SMILES
        3. Remove duplicates
        4. Butina clustering (if enabled)
        5. Train/test split
        6. Calculate similarities
    """
    logger.info("MOLECULAR DATA PREPROCESSING PIPELINE")
    
    stats = {}
    df = input_df.copy()
    stats['original_size'] = len(df)
    logger.info(f"\nOriginal dataset: {len(df)} molecules")
    
    # Step 1: Validate structures
    if validate:
        logger.info(f"\nStep 1: Validating structures...")
        tqdm.pandas(desc="  Validating")
        df['is_valid'] = df[smiles_col].progress_apply(is_valid_structure)
        df = df[df['is_valid'] == True].drop(columns=['is_valid'])
        stats['after_validation'] = len(df)
        logger.info(f"  Valid structures: {len(df)}")
    
    # Step 2: Standardize SMILES
    if standardize:
        logger.info(f"\nStep 2: Standardizing SMILES...")
        tqdm.pandas(desc="  Standardizing")
        df[smiles_col] = df[smiles_col].progress_apply(standardize_and_canonicalize)
        df = df.dropna(subset=[smiles_col])
        stats['after_standardization'] = len(df)
        logger.info(f"  After standardization: {len(df)}")
    
    # Step 3: Remove duplicates
    logger.info(f"\nStep 3: Removing duplicates...")
    df = remove_duplicates(df, smiles_col, label_col)
    stats['after_deduplication'] = len(df)
    
    # Step 4: Butina clustering
    if use_butina:
        logger.info(f"\nStep 4: Butina clustering...")
        df = add_butina_clusters(df, smiles_col)
    
    # Step 5: Train/test split
    logger.info(f"\nStep 5: Train/test split...")
    if use_butina and 'Butina' in df.columns:
        train_df, test_df = split_by_clusters(
            df, test_size=test_size, random_state=random_state
        )
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=test_size, 
            stratify=df[label_col], 
            random_state=random_state
        )
        logger.info(f"  Using random stratified split")
        logger.info(f"  Test size: {len(test_df)} ({100*len(test_df)/len(df):.1f}%)")
        logger.info(f"  Train size: {len(train_df)} ({100*len(train_df)/len(df):.1f}%)")
    
    logger.info(f"\nClass distribution:")
    logger.info(f"  Train: {dict(train_df[label_col].value_counts())}")
    logger.info(f"  Test: {dict(test_df[label_col].value_counts())}")
    
    # Step 6: Calculate similarities
    logger.info(f"\nStep 6: Analyzing train/test similarity...")
    similarities = calculate_train_test_similarity(
        train_df[smiles_col].tolist(),
        test_df[smiles_col].tolist(),
        top_n=5
    )
    
    stats['train_size'] = len(train_df)
    stats['test_size'] = len(test_df)
    stats['mean_similarity'] = float(similarities.mean())
    stats['median_similarity'] = float(np.median(similarities))
    
    logger.info("\nPREPROCESSING COMPLETED")
    logger.info(f"Final train size: {len(train_df)}")
    logger.info(f"Final test size: {len(test_df)}")
    logger.info(f"Mean Tanimoto similarity: {similarities.mean():.3f}")
    
    return train_df, test_df, stats


# ============================================================================
# Convenience Functions
# ============================================================================

def get_preprocessing_statistics(stats: dict) -> pd.DataFrame:
    """Convert stats dictionary to DataFrame for reporting."""
    return pd.DataFrame([stats]).T.rename(columns={0: 'Value'})
