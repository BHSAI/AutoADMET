"""Optuna TPE hyperparameter tuning per descriptor–model pair using stratified CV."""

import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from utils.models import ModelFactory, MetricsCalculator, SMILES_MODELS

import logging

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# Search Spaces
# ============================================================================

def get_search_space(trial, model_name):
    """
    Define Optuna search space for a given model.

    Args:
        trial: Optuna trial object
        model_name: Name of the model

    Returns:
        Dictionary of suggested hyperparameters
    """
    if model_name == 'KNN':
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
        }

    elif model_name == 'SVM':
        return {
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
        }

    elif model_name == 'Bayesian':
        return {
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        }

    elif model_name == 'LogisticRegression':
        return {
            'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
        }

    elif model_name == 'RandomForest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }

    elif model_name == 'LightGBM':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }

    elif model_name == 'XGBoost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }

    else:
        return {}


# Models that have no tunable hyperparameters (or use internal training loops)
SKIP_MODELS = {'TabPFN', 'Chemprop', 'CheMeleon'}


# ============================================================================
# Optimizer
# ============================================================================

class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimizer.

    Uses the same repeated stratified CV as the main pipeline
    to evaluate each trial.
    """

    def __init__(self, n_repeats=5, n_folds=5, random_state=42, device='cpu'):
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.random_state = random_state
        self.device = device

    def _objective(self, trial, X, y, model_name, descriptor_name, metric):
        """Optuna objective: run CV with trial params, return mean metric."""
        params = get_search_space(trial, model_name)
        use_scaler = descriptor_name.lower() == 'mordred'

        scores = []

        for repeat in range(self.n_repeats):
            seed = self.random_state + repeat
            skf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=seed
            )

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if use_scaler:
                    scaler = MinMaxScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)

                try:
                    model = ModelFactory.create(
                        model_name, random_state=seed,
                        device=self.device, **params
                    )
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_val)
                    y_prob = (
                        model.predict_proba(X_val)[:, 1]
                        if hasattr(model, 'predict_proba') else None
                    )

                    metrics = MetricsCalculator.calculate_metrics(
                        y_val, y_pred, y_prob
                    )
                    scores.append(metrics[metric])

                except Exception:
                    scores.append(0.0)

        return float(np.mean(scores))

    def optimize(self, X, y, model_name, descriptor_name,
                 n_trials=50, metric='MCC'):
        """
        Run Optuna optimization for a single model-descriptor pair.

        Args:
            X: Feature matrix
            y: Labels
            model_name: Model to optimize
            descriptor_name: Descriptor name (for scaling logic)
            n_trials: Number of Optuna trials
            metric: Metric to maximize

        Returns:
            Tuple of (best_params dict, best_score)
        """
        if model_name in SKIP_MODELS:
            return {}, None

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(
                trial, X, y, model_name, descriptor_name, metric
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        return study.best_params, study.best_value

    def optimize_all(self, descriptors, y, model_names,
                     n_trials=50, metric='MCC'):
        """
        Optimize hyperparameters for all model-descriptor combinations.

        Args:
            descriptors: Dict of descriptor_name -> feature matrix
            y: Labels
            model_names: List of model names
            n_trials: Number of Optuna trials per combination
            metric: Metric to maximize

        Returns:
            Dict mapping (descriptor, model) -> best_params
        """
        best_params = {}

        for desc_name, X in descriptors.items():
            logger.info(f"\n{desc_name}:")

            for model_name in model_names:
                if desc_name == 'SMILES' and model_name not in SMILES_MODELS:
                    continue
                if desc_name != 'SMILES' and model_name in SMILES_MODELS:
                    continue

                if model_name in SKIP_MODELS:
                    logger.info(f"  {model_name}: skipped (no tunable hyperparameters)")
                    continue

                logger.info(f"  {model_name} ({n_trials} trials)...")
                params, score = self.optimize(
                    X, y, model_name, desc_name,
                    n_trials=n_trials, metric=metric
                )

                best_params[f"{desc_name}_{model_name}"] = params
                logger.info(f"    Best {metric}: {score:.4f}")
                logger.info(f"    Params: {params}")

        return best_params


def save_hyperparameters(params, filepath):
    """Save optimized hyperparameters to JSON."""
    filepath = Path(filepath)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters: {filepath}")


def load_hyperparameters(filepath):
    """Load optimized hyperparameters from JSON."""
    filepath = Path(filepath)
    with open(filepath) as f:
        return json.load(f)
