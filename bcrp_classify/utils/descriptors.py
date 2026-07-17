"""Molecular descriptor generation (Morgan, RDKit, MACCS, Mordred, ChemBERTa, MolFormer) with MD5-based disk caching."""

import pickle
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

from mordred import Calculator, descriptors

import torch
from transformers import AutoTokenizer, AutoModel

import logging

logger = logging.getLogger(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class DescriptorGenerator:
    """
    Generate molecular descriptors from SMILES.
    """
    
    AVAILABLE_DESCRIPTORS = ['Morgan', 'RDKit', 'MACCS', 'Mordred', 'ChemBERTa', 'MolFormer', 'SMILES']
    
    def __init__(self, morgan_radius=2, morgan_nbits=2048, cache_dir=None):
        """
        Initialize descriptor generator.
        
        Args:
            morgan_radius: Morgan fingerprint radius (default: 2 for ECFP4)
            morgan_nbits: Number of bits in fingerprints (default: 2048)
            cache_dir: Directory to cache descriptors (default: None, no caching)
        """
        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Descriptor cache enabled: {self.cache_dir}")
        
        self.rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=morgan_nbits)
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=morgan_radius,
            fpSize=morgan_nbits
        )
        
        self.chemberta_model = None
        self.chemberta_tokenizer = None
        self.molformer_model = None
        self.molformer_tokenizer = None

        
        if self.cache_dir and (self.cache_dir / "mordred_columns.csv").exists():
            self.mordred_columns_ = pd.read_csv(self.cache_dir / "mordred_columns.csv")["column_name"].to_list()
        else:
            # Mordred column names from first clean (used to align new data)
            self.mordred_columns_ = None

        self.device = DEVICE

    def __getstate__(self):
        """Exclude unpicklable RDKit FP generators and torch models."""
        state = self.__dict__.copy()
        # RDKit FingerprintGenerator64 objects cannot be pickled
        state.pop('morgan_gen', None)
        state.pop('rdkit_gen', None)
        # Torch models are large and may not pickle cleanly
        state.pop('chemberta_model', None)
        state.pop('chemberta_tokenizer', None)
        state.pop('molformer_model', None)
        state.pop('molformer_tokenizer', None)
        return state

    def __setstate__(self, state):
        """Recreate FP generators after unpickling."""
        self.__dict__.update(state)
        self.rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=self.morgan_nbits
        )
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius, fpSize=self.morgan_nbits
        )
        self.chemberta_model = None
        self.chemberta_tokenizer = None
        self.molformer_model = None
        self.molformer_tokenizer = None
    
    # ========================================================================
    # Caching Methods
    # ========================================================================
    
    def _create_cache_key(self, descriptor_type, smiles_list, **kwargs):
        """
        Create unique cache key from descriptor type and SMILES.

        Args:
            descriptor_type: Type of descriptor
            smiles_list: List of SMILES strings
            **kwargs: Additional parameters

        Returns:
            Cache filename
        """
        # Use original order for hashing (order matters for row alignment)
        smiles_str = '|'.join(smiles_list)
        
        hash_obj = hashlib.md5(smiles_str.encode())
        smiles_hash = hash_obj.hexdigest()[:16]
        
        params_str = '_'.join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        if params_str:
            filename = f"{descriptor_type}_{params_str}_{smiles_hash}.pkl"
        else:
            filename = f"{descriptor_type}_{smiles_hash}.pkl"
        
        return filename
    
    def _save_to_cache(self, descriptor_type, smiles_list, descriptors, **kwargs):
        """
        Save descriptors to cache.
        
        Args:
            descriptor_type: Type of descriptor
            smiles_list: List of SMILES strings
            descriptors: Generated descriptors
            **kwargs: Additional parameters
        """
        if not self.cache_dir:
            return
        
        cache_key = self._create_cache_key(descriptor_type, smiles_list, **kwargs)
        cache_path = self.cache_dir / cache_key
        
        cache_data = {
            'descriptor_type': descriptor_type,
            'descriptors': descriptors,
            'smiles_list': smiles_list,
            'n_molecules': len(smiles_list),
            'n_features': descriptors.shape[1] if descriptors.ndim > 1 else descriptors.shape[0],
            'parameters': kwargs
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"    Saved to cache: {cache_key}")
    
    def _load_from_cache(self, descriptor_type, smiles_list, **kwargs):
        """
        Load descriptors from cache if available.
        
        Args:
            descriptor_type: Type of descriptor
            smiles_list: List of SMILES strings
            **kwargs: Additional parameters
            
        Returns:
            Cached descriptors or None if not found
        """
        if not self.cache_dir:
            return None
        
        cache_key = self._create_cache_key(descriptor_type, smiles_list, **kwargs)
        cache_path = self.cache_dir / cache_key
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            logger.info(f"    Loaded from cache: {cache_key}")
            return cache_data['descriptors']
        
        except Exception as e:
            logger.info(f"    Warning: Failed to load cache: {e}")
            return None
    
    def clear_cache(self, descriptor_type=None):
        """
        Clear descriptor cache.
        
        Args:
            descriptor_type: If specified, only clear this type. Otherwise clear all.
        """
        if not self.cache_dir:
            logger.info("No cache directory configured")
            return
        
        if descriptor_type:
            pattern = f"{descriptor_type}_*.pkl"
            files = list(self.cache_dir.glob(pattern))
        else:
            files = list(self.cache_dir.glob("*.pkl"))
        
        for file in files:
            file.unlink()
        
        logger.info(f"Cleared {len(files)} cache file(s)")
    
    def get_cache_info(self):
        """
        Get information about cached descriptors.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir:
            return {'enabled': False}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        
        info = {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'total_files': len(cache_files),
            'descriptors': {}
        }
        
        for file in cache_files:
            desc_type = file.stem.split('_')[0]
            info['descriptors'][desc_type] = info['descriptors'].get(desc_type, 0) + 1
        
        return info
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def smiles_to_mols(self, smiles_list):
        """
        Convert SMILES to RDKit molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (valid_mols, valid_indices)
        """
        mols = []
        valid_idx = []
        
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                valid_idx.append(i)
        
        return mols, valid_idx
    
    # ========================================================================
    # Morgan Fingerprints (ECFP4)
    # ========================================================================
    
    def generate_morgan(self, smiles_list):
        """
        Generate Morgan (ECFP) fingerprints.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            numpy array of shape (n_molecules, n_bits)
        """
        mols, _ = self.smiles_to_mols(smiles_list)
        
        fps = []
        for mol in tqdm(mols, desc="    Morgan FPs", leave=False):
            arr = np.zeros(self.morgan_nbits, dtype=np.int8)
            fp = self.morgan_gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)

        return np.array(fps)

    # ========================================================================
    # RDKit Fingerprints
    # ========================================================================

    def generate_rdkit(self, smiles_list):
        """
        Generate RDKit fingerprints.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            numpy array of shape (n_molecules, n_bits)
        """
        mols, _ = self.smiles_to_mols(smiles_list)

        fps = []
        for mol in tqdm(mols, desc="    RDKit FPs", leave=False):
            arr = np.zeros(self.morgan_nbits, dtype=np.int8)
            fp = self.rdkit_gen.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)

        return np.array(fps)

    # ========================================================================
    # MACCS Keys
    # ========================================================================

    def generate_maccs(self, smiles_list):
        """
        Generate MACCS keys fingerprints.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            numpy array of shape (n_molecules, 166)
        """
        mols, _ = self.smiles_to_mols(smiles_list)

        fps = []
        for mol in tqdm(mols, desc="    MACCS keys", leave=False):
            arr = np.zeros(166, dtype=np.int8)
            fp = MACCSkeys.GenMACCSKeys(mol)
            tmp = np.zeros(167, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, tmp)
            arr[:] = tmp[1:]  # Drop bit-0 (unused)
            fps.append(arr)
        
        return np.array(fps)
    
    # ========================================================================
    # Mordred Descriptors
    # ========================================================================
    
    def generate_mordred(self, smiles_list, clean=True):
        """
        Generate Mordred descriptors with optional cleaning.

        On the first call with clean=True, performs full cleaning (remove NaN,
        low variance, high correlation) and saves the surviving column names.
        On subsequent calls, reuses those same columns to guarantee consistent
        feature dimensionality between training and prediction data.

        Args:
            smiles_list: List of SMILES strings
            clean: Whether to clean descriptors (remove NaN, low variance, high correlation)

        Returns:
            numpy array of shape (n_molecules, n_descriptors)
        """
        mols, _ = self.smiles_to_mols(smiles_list)

        calc = Calculator(descriptors, ignore_3D=True)
        desc_df = calc.pandas(pd.Series(mols))

        if clean:
            if self.mordred_columns_ is not None:
                # Reuse columns from the first (training) clean
                logger.info(f"    Aligning Mordred descriptors to {len(self.mordred_columns_)} training columns...")
                # Keep only numeric, replace inf
                desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
                # Select only the training columns; missing ones become NaN
                desc_df = desc_df.reindex(columns=self.mordred_columns_)
                # Fill any NaN with 0 (column may be absent or have NaN for new data)
                desc_df = desc_df.astype(float).fillna(0)
                logger.info(f"    Kept {desc_df.shape[1]} descriptors (aligned to training)")
            else:
                # First call: full cleaning, then store column names
                logger.info(f"    Cleaning Mordred descriptors...")
                original_n = desc_df.shape[1]

                desc_df = desc_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1).select_dtypes(include=[np.number])

                variances = desc_df.var()
                desc_df = desc_df[variances[variances > 0.01].index]

                corr = desc_df.corr(method='spearman').abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = [col for col in upper.columns if (upper[col] > 0.95).any()]
                desc_df = desc_df.drop(columns=to_drop)

                # Save column names for future calls
                self.mordred_columns_ = list(desc_df.columns)
                if self.cache_dir:
                    pd.DataFrame(self.mordred_columns_, columns=["column_name"]).to_csv(self.cache_dir / "mordred_columns.csv")
                logger.info(f"    Kept {desc_df.shape[1]} / {original_n} descriptors")

        return desc_df.values
    
    # ========================================================================
    # ChemBERTa Embeddings
    # ========================================================================
    
    def generate_chemberta(self, smiles_list, batch_size=32):
        """
        Generate ChemBERTa embeddings.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference
            
        Returns:
            numpy array of shape (n_molecules, 768)
        """
        if self.chemberta_model is None:
            logger.info(f"    Loading ChemBERTa model on {self.device}...")
            self.chemberta_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.chemberta_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.chemberta_model.to(self.device)
            self.chemberta_model.eval()
        
        embeddings = []
        n_batches = (len(smiles_list) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), total=n_batches, desc="    ChemBERTa", leave=False):
                batch = smiles_list[i:i+batch_size]

                inputs = self.chemberta_tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(self.device)

                outputs = self.chemberta_model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_emb)

        return np.array(embeddings)

    # ========================================================================
    # MolFormer Embeddings
    # ========================================================================

    def generate_molformer(self, smiles_list, batch_size=32):
        """
        Generate MolFormer embeddings.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for inference

        Returns:
            numpy array of shape (n_molecules, 768)
        """
        if self.molformer_model is None:
            logger.info(f"    Loading MolFormer model on {self.device}...")
            # MolFormer's remote code imports from transformers.onnx, which was
            # removed in transformers>=4.26. Patch in a stub so the import
            # succeeds — ONNX export isn't needed for inference.
            import sys, types
            if "transformers.onnx" not in sys.modules:
                onnx_stub = types.ModuleType("transformers.onnx")
                onnx_stub.OnnxConfig = type("OnnxConfig", (), {})
                sys.modules["transformers.onnx"] = onnx_stub
            self.molformer_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
            self.molformer_model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
            self.molformer_model.to(self.device)
            self.molformer_model.eval()

        embeddings = []
        n_batches = (len(smiles_list) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), total=n_batches, desc="    MolFormer", leave=False):
                batch = smiles_list[i:i+batch_size]

                inputs = self.molformer_tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512
                ).to(self.device)

                outputs = self.molformer_model(**inputs)
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_emb)
        
        return np.array(embeddings)
    
    # ========================================================================
    # Main Interface
    # ========================================================================
    
    def generate(self, descriptor_type, smiles_list, **kwargs):
        """
        Generate descriptors by type with caching support.
        
        Args:
            descriptor_type: One of ['Morgan', 'RDKit', 'MACCS', 'Mordred', 'ChemBERTa', 'MolFormer']
            smiles_list: List of SMILES strings
            **kwargs: Additional arguments for specific descriptor types
            
        Returns:
            numpy array of descriptors
        """
        # SMILES pseudo-descriptor: pass-through for GNN models (no caching needed)
        if descriptor_type == 'SMILES':
            return np.array(smiles_list, dtype=object)

        cached = self._load_from_cache(descriptor_type, smiles_list, **kwargs)
        if cached is not None:
            return cached

        if descriptor_type == 'Morgan':
            result = self.generate_morgan(smiles_list)

        elif descriptor_type == 'RDKit':
            result = self.generate_rdkit(smiles_list)

        elif descriptor_type == 'MACCS':
            result = self.generate_maccs(smiles_list)

        elif descriptor_type == 'Mordred':
            clean = kwargs.get('clean', True)
            # Include number of training columns in cache key so stale caches
            # (generated before column alignment was implemented) are invalidated.
            if self.mordred_columns_ is not None:
                kwargs['_ncols'] = len(self.mordred_columns_)
                # Re-check cache with updated key
                cached = self._load_from_cache(descriptor_type, smiles_list, **kwargs)
                if cached is not None:
                    return cached
            result = self.generate_mordred(smiles_list, clean=clean)

        elif descriptor_type == 'ChemBERTa':
            batch_size = kwargs.get('batch_size', 32)
            result = self.generate_chemberta(smiles_list, batch_size=batch_size)

        elif descriptor_type == 'MolFormer':
            batch_size = kwargs.get('batch_size', 32)
            result = self.generate_molformer(smiles_list, batch_size=batch_size)

        else:
            raise ValueError(f"Unknown descriptor type: {descriptor_type}")

        self._save_to_cache(descriptor_type, smiles_list, result, **kwargs)

        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def get_available_descriptors():
    """Get list of available descriptor types."""
    return DescriptorGenerator.AVAILABLE_DESCRIPTORS



