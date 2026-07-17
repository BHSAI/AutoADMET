"""Model factory, cross-validator, and metrics for all supported classifiers (KNN, SVM, RF, LightGBM, XGBoost, TabPFN, GNNs)."""

import logging
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score
)

import lightgbm as lgb
from xgboost import XGBClassifier

from tabpfn import TabPFNClassifier

import torch

logger = logging.getLogger(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Suppress noisy Lightning logs
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

# Models that take SMILES directly (no descriptor matrix)
SMILES_MODELS = {'Chemprop', 'CheMeleon'}

# ============================================================================
# GNN Wrappers (sklearn-compatible)
# ============================================================================

class ChempropClassifier:
    """
    Sklearn-compatible wrapper around Chemprop's D-MPNN for binary classification.

    Takes SMILES strings directly (no descriptor matrix needed).
    Internally converts to molecular graphs and trains via PyTorch Lightning.
    """

    def __init__(self, max_epochs=20, accelerator=None, random_state=42):
        self.max_epochs = max_epochs
        self.accelerator = accelerator or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.model = None
        self.trainer = None

    def _build_model(self):
        """Build a fresh D-MPNN model."""
        from chemprop import nn as chemprop_nn, models as chemprop_models

        mp = chemprop_nn.BondMessagePassing()
        agg = chemprop_nn.MeanAggregation()
        ffn = chemprop_nn.BinaryClassificationFFN(n_tasks=1)
        return chemprop_models.MPNN(mp, agg, ffn)

    @staticmethod
    def _make_dataloader(smiles, y=None, shuffle=False):
        """Convert SMILES (and optional labels) to a chemprop DataLoader."""
        from chemprop import data as chemprop_data, featurizers

        if y is not None:
            datapoints = [
                chemprop_data.MoleculeDatapoint.from_smi(smi, [float(yi)])
                for smi, yi in zip(smiles, y)
            ]
        else:
            datapoints = [
                chemprop_data.MoleculeDatapoint.from_smi(smi)
                for smi in smiles
            ]

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        dataset = chemprop_data.MoleculeDataset(datapoints, featurizer)
        import os
        n_workers = min(os.cpu_count() or 0, 4)
        return chemprop_data.build_dataloader(dataset, num_workers=n_workers, shuffle=shuffle)

    def fit(self, X, y):
        """
        Train the D-MPNN model.

        Args:
            X: numpy array of SMILES strings
            y: numpy array of binary labels
        """
        import lightning.pytorch as pl
        import torch

        torch.set_float32_matmul_precision("medium")
        pl.seed_everything(self.random_state, workers=True)

        smiles = list(X)
        y_arr = np.array(y)

        # 90/10 train/val split for Lightning Trainer
        try:
            tr_smi, val_smi, y_tr, y_val = train_test_split(
                smiles, y_arr, test_size=0.1, stratify=y_arr,
                random_state=self.random_state,
            )
        except ValueError:
            # Fallback without stratification for very small/imbalanced datasets
            tr_smi, val_smi, y_tr, y_val = train_test_split(
                smiles, y_arr, test_size=0.1,
                random_state=self.random_state,
            )

        train_loader = self._make_dataloader(tr_smi, y_tr, shuffle=True)
        val_loader = self._make_dataloader(val_smi, y_val)

        self.model = self._build_model()
        self.trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator=self.accelerator,
            devices=1,
            max_epochs=self.max_epochs,
        )
        self.trainer.fit(self.model, train_loader, val_loader)
        return self

    def __getstate__(self):
        """Exclude unpicklable Lightning Trainer from serialization."""
        state = self.__dict__.copy()
        state['trainer'] = None
        return state

    def _ensure_trainer(self):
        """Lazily create a Trainer for prediction (e.g. after unpickling)."""
        if self.trainer is None:
            import lightning.pytorch as pl
            self.trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                accelerator=self.accelerator,
                devices=1,
            )

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: numpy array of SMILES strings

        Returns:
            (n, 2) array of [P(class=0), P(class=1)]
        """
        self._ensure_trainer()
        loader = self._make_dataloader(list(X))

        with torch.inference_mode():
            preds = self.trainer.predict(self.model, loader)

        # BinaryClassificationFFN applies sigmoid internally
        proba_1 = np.concatenate([p.cpu().numpy() for p in preds]).flatten()
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class CheMeleonClassifier(ChempropClassifier):
    """
    Sklearn-compatible wrapper around CheMeleon (pretrained D-MPNN).

    Uses pretrained message-passing weights from Zenodo, then fine-tunes
    on the target task. Especially effective for small datasets.
    """

    WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    _CACHE_DIR = Path.home() / '.cache' / 'chemeleon'
    _WEIGHTS_FILE = _CACHE_DIR / 'chemeleon_mp.pt'

    def _download_weights(self):
        """Download pretrained weights if not cached."""
        if not self._WEIGHTS_FILE.exists():
            self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("    Downloading CheMeleon pretrained weights...")
            from urllib.request import urlretrieve
            urlretrieve(self.WEIGHTS_URL, self._WEIGHTS_FILE)
            logger.info(f"    Saved to {self._WEIGHTS_FILE}")

    def _build_model(self):
        """Build MPNN with CheMeleon pretrained message-passing weights."""
        from chemprop import nn as chemprop_nn, models as chemprop_models

        self._download_weights()

        chemeleon_state = torch.load(
            self._WEIGHTS_FILE, weights_only=True, map_location='cpu'
        )
        mp = chemprop_nn.BondMessagePassing(**chemeleon_state['hyper_parameters'])
        mp.load_state_dict(chemeleon_state['state_dict'])

        agg = chemprop_nn.MeanAggregation()
        ffn = chemprop_nn.BinaryClassificationFFN(
            n_tasks=1,
            input_dim=mp.output_dim,
        )

        return chemprop_models.MPNN(mp, agg, ffn, batch_norm=False)


# ============================================================================
# Model Factory
# ============================================================================

class ModelFactory:
    """Factory for creating ML models."""
    
    AVAILABLE_MODELS = ['KNN', 'SVM', 'Bayesian', 'LogisticRegression', 'RandomForest',
                        'LightGBM', 'XGBoost', 'TabPFN', 'Chemprop', 'CheMeleon']
    
    @staticmethod
    def create_knn(**kwargs):
        """Create K-Nearest Neighbors classifier."""
        defaults = {'n_neighbors': 5}
        params = {**defaults, **kwargs}
        return KNeighborsClassifier(**params)

    @staticmethod
    def create_svm(random_state=42, **kwargs):
        """Create Support Vector Machine classifier."""
        defaults = {
            'kernel': 'rbf',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': random_state,
        }
        params = {**defaults, **kwargs}
        return SVC(**params)

    @staticmethod
    def create_bayesian(**kwargs):
        """Create Bayesian (BernoulliNB) classifier."""
        defaults = {'alpha': 1.0}
        params = {**defaults, **kwargs}
        return BernoulliNB(**params)

    @staticmethod
    def create_logistic_regression(random_state=42, **kwargs):
        """Create Logistic Regression classifier."""
        defaults = {
            'max_iter': 2000,
            'class_weight': 'balanced',
            'random_state': random_state,
        }
        params = {**defaults, **kwargs}
        return LogisticRegression(**params)

    @staticmethod
    def create_random_forest(random_state=42, **kwargs):
        """Create Random Forest classifier."""
        defaults = {
            'n_estimators': 500,
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': random_state,
        }
        params = {**defaults, **kwargs}
        return RandomForestClassifier(**params)

    @staticmethod
    def create_lightgbm(random_state=42, **kwargs):
        """Create LightGBM classifier."""
        defaults = {
            'n_estimators': 500,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1,
        }
        params = {**defaults, **kwargs}
        return lgb.LGBMClassifier(**params)

    @staticmethod
    def create_xgboost(random_state=42, **kwargs):
        """Create XGBoost classifier."""
        defaults = {
            'n_estimators': 500,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
        }
        params = {**defaults, **kwargs}
        return XGBClassifier(**params)

    @staticmethod
    def create_tabpfn(device='cpu', random_state=42, **kwargs):
        """Create TabPFN classifier."""
        return TabPFNClassifier(device=device, random_state=random_state, ignore_pretraining_limits=True)

    @staticmethod
    def create_chemprop(random_state=42, **kwargs):
        """Create Chemprop D-MPNN classifier."""
        return ChempropClassifier(random_state=random_state, **kwargs)

    @staticmethod
    def create_chemeleon(random_state=42, **kwargs):
        """Create CheMeleon (pretrained D-MPNN) classifier."""
        return CheMeleonClassifier(random_state=random_state, **kwargs)

    @classmethod
    def create(cls, model_name, random_state=42, device='cpu', **kwargs):
        """
        Create model by name.

        Args:
            model_name: One of AVAILABLE_MODELS
            random_state: Random seed
            device: Device for TabPFN ('cpu' or 'cuda')
            **kwargs: Additional model-specific arguments

        Returns:
            Scikit-learn compatible model
        """
        factories = {
            'KNN': cls.create_knn,
            'SVM': cls.create_svm,
            'Bayesian': cls.create_bayesian,
            'LogisticRegression': cls.create_logistic_regression,
            'RandomForest': cls.create_random_forest,
            'LightGBM': cls.create_lightgbm,
            'XGBoost': cls.create_xgboost,
            'TabPFN': cls.create_tabpfn,
            'Chemprop': cls.create_chemprop,
            'CheMeleon': cls.create_chemeleon,
        }

        if model_name not in factories:
            raise ValueError(f"Unknown model: {model_name}")

        if model_name == 'TabPFN':
            return factories[model_name](device=device, random_state=random_state, **kwargs)
        elif model_name in ['KNN', 'Bayesian']:
            return factories[model_name](**kwargs)
        else:
            return factories[model_name](random_state=random_state, **kwargs)


# ============================================================================
# Metrics Calculator
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive classification metrics."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC-AUC and PR-AUC)
            
        Returns:
            Dictionary of metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        gmean = np.sqrt(sensitivity * specificity)
        
        metrics = {
            'ROC_AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0,
            'PR_AUC': average_precision_score(y_true, y_prob) if y_prob is not None else 0.0,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'GMean': gmean,
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'Kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics


# ============================================================================
# Cross-Validation Trainer
# ============================================================================

class CrossValidator:
    """
    Handle repeated stratified cross-validation.
    
    Usage:
        cv = CrossValidator(n_repeats=5, n_folds=5)
        results = cv.run_cv(X, y, 'XGBoost', 'Morgan')
    """
    
    def __init__(self, n_repeats=5, n_folds=5, random_state=42):
        """
        Initialize cross-validator.
        
        Args:
            n_repeats: Number of CV repeats
            n_folds: Number of folds per repeat
            random_state: Base random seed
        """
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.random_state = random_state
        self.device = DEVICE
    
    def run_cv(self, X, y, model_name, descriptor_name, model_params=None):
        """
        Run repeated stratified K-fold CV.

        Args:
            X: Feature matrix
            y: Labels
            model_name: Name of model to train
            descriptor_name: Name of descriptor (for recording)
            model_params: Optional dict of hyperparameters to pass to ModelFactory

        Returns:
            List of dictionaries with per-fold results
        """
        if model_params is None:
            model_params = {}

        results = []

        use_scaler = descriptor_name.lower() == 'mordred'
        if use_scaler:
            logger.info(f"    Using MinMaxScaler for {descriptor_name} (fit per fold)")
        else:
            logger.info(f"    No scaling for {descriptor_name} (already normalized)")

        total_folds = self.n_repeats * self.n_folds
        pbar = tqdm(total=total_folds, desc=f"    {model_name}", leave=False)

        for repeat in range(self.n_repeats):
            seed = self.random_state + repeat
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Scale inside the fold to prevent data leakage
                if use_scaler:
                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                try:
                    model = ModelFactory.create(
                        model_name, random_state=seed,
                        device=self.device, **model_params
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

                    metrics = MetricsCalculator.calculate_metrics(y_val, y_pred, y_prob)

                    result = {
                        'Descriptor': descriptor_name,
                        'Model': model_name,
                        'Repeat': repeat + 1,
                        'Fold': fold + 1,
                        **metrics
                    }
                    results.append(result)

                except Exception as e:
                    logger.info(f"      Error in fold {fold+1}: {e}")

                pbar.update(1)

        pbar.close()
        return results
    


# ============================================================================
# Convenience Functions
# ============================================================================

def get_available_models():
    """Get list of available model names."""
    return ModelFactory.AVAILABLE_MODELS


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    return MetricsCalculator.calculate_metrics(y_true, y_pred, y_prob)
