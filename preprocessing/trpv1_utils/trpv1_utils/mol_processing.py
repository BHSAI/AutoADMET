"""
Molecular validation and standardization utilities.

This module provides functions for:
- Validating SMILES strings (organic molecules, element filtering)
- RDKit-based molecular standardization (cleanup, charge, tautomer, stereo, isotope)
- InChIKey generation
- Complete SMILES processing pipeline
"""

import logging
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize as rms
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from .config import ELEMENT_WHITELIST

# ============================================================================
# Module-level Constants
# ============================================================================

# SMARTS pattern for carbon (to filter organic molecules)
_CARBON = Chem.MolFromSmarts("[#6]")

# RDKit standardization objects (reused for efficiency)
_TAUT_ENUM = rms.TautomerEnumerator()
_UNCHARGER = rms.Uncharger()

# ============================================================================
# Validation Functions
# ============================================================================

def is_valid_mol(smiles):
    """
    Validate and sanitize a SMILES string.

    Checks:
    1. Is a non-empty string
    2. Contains at least one carbon atom (organic molecule)
    3. Contains only whitelisted elements
    4. Can be sanitized by RDKit

    Args:
        smiles: SMILES string to validate

    Returns:
        RDKit Mol object if valid, None otherwise
    """
    # Validate input type and handle edge cases
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()
    if smiles == "" or smiles.lower() in {"nan", "na", "."}:
        return None

    # Parse SMILES without sanitization (to check structure first)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if not mol:
        return None

    # Check for carbon (organic molecule requirement)
    if not mol.HasSubstructMatch(_CARBON):
        return None

    # Check for only whitelisted elements
    if any(a.GetSymbol() not in ELEMENT_WHITELIST for a in mol.GetAtoms()):
        return None

    # Attempt sanitization
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        logging.debug(f"Sanitization failed for {smiles}: {e}")
        return None


# ============================================================================
# Standardization Functions
# ============================================================================

def standardize_mol(mol):
    """
    Apply RDKit standardization pipeline to a molecule.

    Steps:
    1. InChI round-trip (canonicalize)
    2. Cleanup (fix valence, aromaticity, etc.)
    3. Fragment parent (keep largest fragment)
    4. Uncharge (neutralize)
    5. Charge parent (ensure integral charge)
    6. Isotope parent (strip isotopes)
    7. Stereo parent (normalize stereochemistry)
    8. Tautomer canonicalization (pick canonical tautomer)

    Args:
        mol: RDKit Mol object

    Returns:
        Canonical isomeric SMILES string, or None if standardization fails
    """
    try:
        # InChI round-trip for initial canonicalization
        mol = Chem.MolFromInchi(Chem.MolToInchi(mol))

        # Apply standardization steps
        mol = rms.Cleanup(mol)                 # fix valence, aromaticity, etc.
        mol = rms.FragmentParent(mol)          # get largest fragment
        mol = _UNCHARGER.uncharge(mol)         # neutralize charges
        mol = rms.ChargeParent(mol)            # ensure integral charge
        mol = rms.IsotopeParent(mol)           # strip isotope information
        mol = rms.StereoParent(mol)            # normalize stereochemistry
        mol = _TAUT_ENUM.Canonicalize(mol)     # pick canonical tautomer

        # Return canonical isomeric SMILES
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    except Exception as e:
        logging.debug(f"Standardization failed: {e}")
        return None


# ============================================================================
# InChIKey Functions
# ============================================================================

def to_inchikey(smiles):
    """
    Generate InChIKey from SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        InChIKey string, or None if generation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi = MolToInchi(mol)
        return InchiToInchiKey(inchi)
    except Exception as e:
        logging.debug(f"InChIKey generation failed for {smiles}: {e}")
        return None


# ============================================================================
# Combined Processing Pipeline
# ============================================================================

def process_smiles(smiles):
    """
    Complete SMILES processing pipeline.

    Steps:
    1. Validate molecule (is_valid_mol)
    2. Standardize (standardize_mol)
    3. Generate InChIKey (to_inchikey)

    Args:
        smiles: Input SMILES string

    Returns:
        Tuple of (canonical_smiles, inchikey), or (None, None) if processing fails
    """
    # Validate molecule
    mol = is_valid_mol(smiles)
    if not mol:
        return None, None

    # Standardize
    canon = standardize_mol(mol)
    if not canon:
        return None, None

    # Generate InChIKey
    ikey = to_inchikey(canon)

    return canon, ikey


# ============================================================================
# Stereochemistry and Tautomer Utilities
# ============================================================================

def nostereo_tautomer(smiles):
    """
    Generate canonical SMILES without stereochemistry.

    This is used for detecting duplicates that differ only in stereochemistry.

    Args:
        smiles: Input SMILES string

    Returns:
        Non-isomeric canonical SMILES (no stereo), or None if processing fails
    """
    if not isinstance(smiles, str):
        return None

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if not mol:
            return None

        # Remove stereochemistry
        Chem.RemoveStereochemistry(mol)

        # Canonicalize tautomer
        mol = _TAUT_ENUM.Canonicalize(mol)

        # Return non-isomeric SMILES (no stereo markers)
        return Chem.MolToSmiles(mol, isomericSmiles=False)

    except Exception:
        return None


def extract_inchikey14(inchikey):
    """
    Extract first 14 characters of InChIKey (connectivity layer).

    The InChIKey format is: XXXXXXXXXXXXXX-YYYYYYYYYY-Z
    - First 14 chars (X): connectivity layer (structure without stereo/isotope)
    - Next 10 chars (Y): stereochemistry layer
    - Last char (Z): protonation

    Args:
        inchikey: Full InChIKey string

    Returns:
        First 14 characters (connectivity layer), or None if invalid
    """
    if not isinstance(inchikey, str):
        return None
    return inchikey.split("-")[0] if "-" in inchikey else None
