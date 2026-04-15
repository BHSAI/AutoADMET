"""
Molecular fingerprint generation utilities.

This module provides functions for generating various molecular fingerprints:
- RDKit fingerprints (2048-bit)
- Morgan (ECFP) fingerprints (2048-bit, radius=2)
- MACCS keys (166-bit)
- Atom Pair fingerprints (2048-bit)
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from .config import MORGAN_RADIUS, MORGAN_FP_SIZE

# ============================================================================
# Fingerprint Generators (Module-level, reused for efficiency)
# ============================================================================

# RDKit fingerprint generator (2048-bit)
_rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=MORGAN_FP_SIZE)

# Morgan fingerprint generator (ECFP4, 2048-bit, radius=2)
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_FP_SIZE
)

# ============================================================================
# Fingerprint Generation Functions
# ============================================================================

def generate_rdkit_fp(mol: Chem.Mol) -> np.ndarray:
    """
    Generate 2048-bit RDKit fingerprint from a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        numpy array of shape (2048,) with dtype int8
    """
    arr = np.zeros(MORGAN_FP_SIZE, dtype=np.int8)
    if mol is not None:
        fp = _rdkit_gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def generate_morgan_fp(mol: Chem.Mol) -> np.ndarray:
    """
    Generate 2048-bit Morgan (ECFP4) fingerprint from a molecule.

    Uses radius=2 (equivalent to ECFP4).

    Args:
        mol: RDKit Mol object

    Returns:
        numpy array of shape (2048,) with dtype int8
    """
    arr = np.zeros(MORGAN_FP_SIZE, dtype=np.int8)
    if mol is not None:
        fp = _morgan_gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def generate_maccs_fp(mol: Chem.Mol) -> np.ndarray:
    """
    Generate 166-bit MACCS keys fingerprint from a molecule.

    Note: The original MACCS is 167 bits, but bit-0 is unused.
    This function drops bit-0 and returns bits 1-166.

    Args:
        mol: RDKit Mol object

    Returns:
        numpy array of shape (166,) with dtype int8
    """
    tmp = np.zeros(167, dtype=np.int8)
    arr = np.zeros(166, dtype=np.int8)
    if mol is not None:
        fp = MACCSkeys.GenMACCSKeys(mol)
        DataStructs.ConvertToNumpyArray(fp, tmp)
        arr[:] = tmp[1:]  # Drop bit-0 (unused)
    return arr


# ============================================================================
# Fingerprint Registry
# ============================================================================

# Dictionary mapping fingerprint names to their generator functions
FINGERPRINT_GENERATORS = {
    "RDKITfp": generate_rdkit_fp,
    "Morgan": generate_morgan_fp,
    "MACCS": generate_maccs_fp,
}


def get_fingerprint_generator(fingerprint_type: str):
    """
    Get fingerprint generator function by name.

    Args:
        fingerprint_type: Name of fingerprint type
            ('RDKITfp', 'Morgan', 'MACCS')

    Returns:
        Fingerprint generator function

    Raises:
        ValueError: If fingerprint type is unknown
    """
    if fingerprint_type not in FINGERPRINT_GENERATORS:
        raise ValueError(
            f"Unknown fingerprint type: {fingerprint_type}. "
            f"Available: {list(FINGERPRINT_GENERATORS.keys())}"
        )
    return FINGERPRINT_GENERATORS[fingerprint_type]


def generate_fingerprints(mols: list, fingerprint_type: str) -> np.ndarray:
    """
    Generate fingerprints for a list of molecules.

    Args:
        mols: List of RDKit Mol objects
        fingerprint_type: Type of fingerprint to generate
            ('RDKITfp', 'Morgan', 'MACCS')

    Returns:
        numpy array of shape (n_molecules, fingerprint_size)
    """
    fp_func = get_fingerprint_generator(fingerprint_type)

    # Generate fingerprints
    fps = [fp_func(mol) for mol in mols]

    return np.array(fps)


# ============================================================================
# Molecular Conversion Utilities
# ============================================================================

def smiles_to_mols(smiles_list: list, return_indices: bool = False):
    """
    Convert list of SMILES to RDKit Mol objects.

    Args:
        smiles_list: List of SMILES strings
        return_indices: If True, also return indices of valid molecules

    Returns:
        If return_indices=False: list of Mol objects (None for invalid SMILES)
        If return_indices=True: tuple of (list of valid Mols, list of valid indices)
    """
    if not return_indices:
        return [Chem.MolFromSmiles(smi) for smi in smiles_list]

    # Return only valid molecules with their indices
    mols = []
    indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            indices.append(i)

    return mols, indices
