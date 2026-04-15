"""
SHAP analysis utilities for TRPV1 ML benchmark.

Provides functions for:
- Training models and computing SHAP values
- Creating SHAP visualizations (beeswarm, bar plots)
- Morgan bit analysis and visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

def check_shap_available():
    """Check if shap package is available."""
    try:
        import shap
        return True
    except ImportError:
        return False

def featurize_smiles_to_morgan_df(smiles_series: pd.Series,
                                   radius: int = 2,
                                   n_bits: int = 2048) -> pd.DataFrame:
    """
    Convert SMILES to Morgan fingerprint DataFrame.

    Args:
        smiles_series: Series of SMILES strings
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint

    Returns:
        DataFrame with columns Bit_0 ... Bit_{n_bits-1}
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    arrs = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(n_bits, dtype=int)
        if mol is not None:
            fp = mgen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
        arrs.append(arr)

    X = pd.DataFrame(arrs, columns=[f"Bit_{i}" for i in range(n_bits)])
    return X

def compute_shap_values(model, X_background, X_test,
                       background_sample_size: int = 300,
                       random_state: int = 42):
    """
    Compute SHAP values using LinearExplainer.

    Args:
        model: Trained sklearn model
        X_background: Background data for SHAP explainer
        X_test: Test data to explain
        background_sample_size: Max background samples for speed
        random_state: Random seed

    Returns:
        SHAP values matrix (n_samples, n_features)
    """
    import shap

    # Subsample background if needed
    if len(X_background) > background_sample_size:
        background = shap.sample(X_background, background_sample_size,
                                random_state=random_state)
    else:
        background = X_background.copy()

    # Create explainer
    explainer = shap.LinearExplainer(
        model,
        background,
        feature_perturbation="interventional"
    )

    # Compute SHAP values
    shap_vals = explainer.shap_values(X_test)

    # Handle old/new SHAP API variants
    if isinstance(shap_vals, list):
        # SHAP might return one array per class; take the positive class
        shap_matrix = shap_vals[1]
    else:
        shap_matrix = shap_vals

    assert shap_matrix.ndim == 2, "Expected 2D SHAP matrix"

    return shap_matrix

def create_shap_summary_plot(shap_values, X_data,
                            max_display: int = 20,
                            save_path: Optional[Path] = None):
    """
    Create SHAP beeswarm summary plot.

    Args:
        shap_values: SHAP values matrix
        X_data: Feature data
        max_display: Maximum features to display
        save_path: Path to save figure (None = don't save)
    """
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_data,
        max_display=max_display,
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def create_shap_bar_plot(shap_values, X_data,
                        max_display: int = 20,
                        save_path: Optional[Path] = None):
    """
    Create SHAP bar plot (mean |SHAP| values).

    Args:
        shap_values: SHAP values matrix
        X_data: Feature data
        max_display: Maximum features to display
        save_path: Path to save figure (None = don't save)
    """
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_data,
        max_display=max_display,
        plot_type="bar",
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def get_top_shap_features(shap_values, feature_names, n_top: int = 20) -> List[Tuple[str, float]]:
    """
    Get top features by mean absolute SHAP value.

    Args:
        shap_values: SHAP values matrix (n_samples, n_features)
        feature_names: List of feature names
        n_top: Number of top features to return

    Returns:
        List of (feature_name, mean_abs_shap) tuples
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    top_indices = np.argsort(mean_abs_shap)[::-1][:n_top]
    top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_indices]

    return top_features

# ============================================================================
# Morgan Bit Visualization
# ============================================================================

def get_atoms_for_morgan_bit(mol, bit_id: int,
                             radius: int = 2,
                             n_bits: int = 2048) -> List[int]:
    """
    Get atom indices associated with a Morgan fingerprint bit.

    Args:
        mol: RDKit Mol object
        bit_id: Morgan bit index
        radius: Morgan fingerprint radius
        n_bits: Number of bits

    Returns:
        List of atom indices
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    bitInfo = {}
    _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bitInfo
    )

    if bit_id not in bitInfo:
        return []

    atoms = set()
    for center, rad in bitInfo[bit_id]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
        for bond_idx in env:
            bond = mol.GetBondWithIdx(bond_idx)
            atoms.add(bond.GetBeginAtomIdx())
            atoms.add(bond.GetEndAtomIdx())

    return list(atoms)

def draw_mol_with_highlight(mol, highlight_atoms: List[int],
                           out_path: Path,
                           legend: str = "",
                           img_size: Tuple[int, int] = (400, 300)):
    """
    Draw molecule with highlighted atoms to SVG file.

    Args:
        mol: RDKit Mol object
        highlight_atoms: List of atom indices to highlight
        out_path: Output SVG file path
        legend: Legend text for the drawing
        img_size: Image size (width, height)
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(img_size[0], img_size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        legend=legend
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    out_path.write_text(svg, encoding="utf-8")

def calculate_bit_frequency(mols: List, bit_id: int,
                           radius: int = 2,
                           n_bits: int = 2048) -> Tuple[int, float]:
    """
    Calculate frequency of a Morgan bit in a set of molecules.

    Args:
        mols: List of RDKit Mol objects
        bit_id: Morgan bit index
        radius: Morgan fingerprint radius
        n_bits: Number of bits

    Returns:
        Tuple of (count, frequency) where frequency is in [0, 1]
    """
    from rdkit.Chem import rdMolDescriptors

    count = 0
    valid_mols = 0

    for mol in mols:
        if mol is None:
            continue
        valid_mols += 1

        bitInfo = {}
        _ = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, bitInfo=bitInfo
        )

        if bit_id in bitInfo:
            count += 1

    frequency = count / valid_mols if valid_mols > 0 else 0.0

    return count, frequency

def find_example_molecules_for_bit(mols: List, bit_id: int,
                                  max_examples: int = 3,
                                  radius: int = 2,
                                  n_bits: int = 2048) -> List[Tuple[int, List[int]]]:
    """
    Find example molecules containing a specific Morgan bit.

    Args:
        mols: List of RDKit Mol objects
        bit_id: Morgan bit index
        max_examples: Maximum number of examples to return
        radius: Morgan fingerprint radius
        n_bits: Number of bits

    Returns:
        List of (molecule_index, highlighted_atoms) tuples
    """
    examples = []

    for idx, mol in enumerate(mols):
        if mol is None:
            continue

        hl_atoms = get_atoms_for_morgan_bit(mol, bit_id, radius, n_bits)

        if hl_atoms:
            examples.append((idx, hl_atoms))

            if len(examples) >= max_examples:
                break

    return examples
