#!/usr/bin/env python
"""Train the best CV model (or stacking ensemble) on all training data and evaluate on the test set.

Can also load a previously saved model and evaluate it on a new test set without retraining
(--load-model flag).
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import Config
from utils.descriptors import DescriptorGenerator
from utils.models import ModelFactory, calculate_metrics
from utils.ensemble import StackingEnsemble, select_top_k

warnings.filterwarnings('ignore')

import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def find_best_model(results_dir, metric='MCC'):
    """Find best model/descriptor combination from CV results."""
    results_file = Path(results_dir) / 'results_summary.csv'
    
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            "Run cross-validation first with pipeline.py"
        )
    
    df = pd.read_csv(results_file)
    
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not in results. Available: {df.columns.tolist()}"
        )
    
    best_idx = df[metric].idxmax()
    best_row = df.loc[best_idx]
    
    descriptor = best_row['Descriptor']
    model = best_row['Model']
    
    logger.info(f"\nBEST MODEL SELECTION (by {metric})")
    logger.info(f"  Descriptor: {descriptor}")
    logger.info(f"  Model: {model}")
    logger.info(f"  CV {metric}: {best_row[metric]:.4f}")
    
    return descriptor, model


def load_and_validate_data(filepath, dataset_name="Data"):
    """Load and validate SMILES data with labels."""
    logger.info(f"\nLOADING {dataset_name.upper()}")
    
    df = pd.read_csv(filepath)
    logger.info(f"  Total samples: {len(df)}")
    
    if 'SMILES' not in df.columns or 'CLASS' not in df.columns:
        raise ValueError(
            f"CSV must have 'SMILES' and 'CLASS' columns. "
            f"Found: {df.columns.tolist()}"
        )
    
    valid_smiles = []
    valid_labels = []
    
    for smi, label in tqdm(zip(df['SMILES'], df['CLASS']), total=len(df), desc="  Validating SMILES", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_labels.append(label)

    logger.info(f"  Valid SMILES: {len(valid_smiles)}")
    logger.info(f"  Invalid SMILES: {len(df) - len(valid_smiles)}")
    
    if len(valid_smiles) == 0:
        raise ValueError(f"No valid SMILES found in {filepath}")
    
    labels = np.array(valid_labels)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    logger.info(f"  Class 0: {n_neg} ({100*n_neg/len(labels):.1f}%)")
    logger.info(f"  Class 1: {n_pos} ({100*n_pos/len(labels):.1f}%)")
    
    return valid_smiles, labels


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train final model and evaluate on test set (single step)",
    )
    
    parser.add_argument('--train', default=None,
                        help='Training data CSV (SMILES, CLASS) — not needed with --load-model')
    parser.add_argument('--test', required=True,
                        help='Test data CSV (SMILES, CLASS)')
    parser.add_argument('--output', required=True,
                        help='Output directory')

    parser.add_argument('--load-model', default=None, metavar='DIR',
                        help='Load a previously saved model directory and evaluate on --test '
                             'without retraining (single model or ensemble)')

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--results',
                       help='CV results directory (auto-select best model)')
    group.add_argument('--descriptor',
                       help='Descriptor type (manual selection)')
    
    parser.add_argument('--model',
                        help='Model type (required with --descriptor)')
    parser.add_argument('--metric', default='MCC',
                        help='Metric for best model selection (default: MCC)')
    parser.add_argument('--hyperparams',
                        help='Path to optimized_hyperparameters.json from Optuna')

    parser.add_argument('--ensemble', action='store_true',
                        help='Use stacking ensemble of top-K models (requires --results)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Number of top models for ensemble. With --load-model, '
                             'verifies the saved ensemble has exactly this many base models.')

    args = parser.parse_args()

    if args.load_model:
        if args.results or args.descriptor or args.ensemble:
            parser.error("--load-model cannot be combined with --results, --descriptor, or --ensemble")
    else:
        if not args.train:
            parser.error("--train is required unless --load-model is used")
        if not args.results and not args.descriptor:
            parser.error("one of --results or --descriptor is required unless --load-model is used")
        if args.ensemble and not args.results:
            parser.error("--ensemble requires --results")
        if args.ensemble and args.top_k is None:
            args.top_k = 5
        if args.descriptor and not args.model:
            parser.error("--model required with --descriptor")
        if args.model and not args.descriptor:
            parser.error("--descriptor required with --model")

    start_time = datetime.now()

    try:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        test_smiles, y_test = load_and_validate_data(args.test, "Test Data")

        if args.load_model:
            logger.info("\nLOAD MODEL AND EVALUATE ON TEST SET")
            logger.info(f"Started: {start_time:%Y-%m-%d %H:%M:%S}")
            _run_load_model(args, output_dir, test_smiles, y_test, start_time)
        else:
            train_smiles, y_train = load_and_validate_data(args.train, "Training Data")
            if args.ensemble:
                logger.info("\nSTACKING ENSEMBLE: TRAIN AND EVALUATE")
                logger.info(f"Started: {start_time:%Y-%m-%d %H:%M:%S}")
                _run_ensemble(args, output_dir, train_smiles, y_train,
                              test_smiles, y_test, start_time)
            else:
                logger.info("\nTRAIN FINAL MODEL AND EVALUATE ON TEST SET")
                logger.info(f"Started: {start_time:%Y-%m-%d %H:%M:%S}")
                _run_single_model(args, output_dir, train_smiles, y_train,
                                  test_smiles, y_test, start_time)

    except Exception as e:
        logger.info(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def _run_load_model(args, output_dir, test_smiles, y_test, start_time):
    """Evaluate a previously saved single model or ensemble on a new test set."""
    load_dir = Path(args.load_model)

    ensemble_meta = load_dir / 'ensemble_metadata.json'
    single_meta   = load_dir / 'model_metadata.json'
    cache_dir     = load_dir / 'descriptor_cache'

    if ensemble_meta.exists():
        with open(ensemble_meta) as f:
            meta = json.load(f)

        saved_top_k = meta.get('top_k')
        if saved_top_k is not None:
            ensemble_file = load_dir / f'stacking_ensemble_top{saved_top_k}.pkl'
        else:
            # Fallback: find any stacking_ensemble*.pkl in the directory
            candidates = sorted(load_dir.glob('stacking_ensemble*.pkl'))
            if not candidates:
                raise FileNotFoundError(f"No stacking_ensemble*.pkl found in {load_dir}")
            ensemble_file = candidates[0]
        if not ensemble_file.exists():
            raise FileNotFoundError(f"Ensemble file not found: {ensemble_file}")

        logger.info(f"\nLoading ensemble from {ensemble_file.name}")
        from utils.ensemble import StackingEnsemble
        ensemble: StackingEnsemble = joblib.load(ensemble_file)

        if args.top_k is not None:
            n_loaded = len(ensemble.base_configs)
            if n_loaded != args.top_k:
                base_labels = [f"{d}+{m}" for d, m in ensemble.base_configs]
                raise ValueError(
                    f"--top-k {args.top_k} does not match the loaded ensemble "
                    f"(has {n_loaded} base models: {base_labels}). "
                    f"Either omit --top-k or set --top-k {n_loaded}."
                )

        ensemble.desc_gen = DescriptorGenerator(cache_dir=cache_dir)
        ensemble.descriptors_train = {}
        ensemble.descriptors_test  = {}

        base_labels = [f"{d}+{m}" for d, m in ensemble.base_configs]
        logger.info(f"  Base models: {base_labels}")

        # Load descriptors from cache so as not to include in prediction time
        for descriptor in set({descriptor for (descriptor, _) in ensemble.base_configs}):
            ensemble.descriptors_test[descriptor] = ensemble.desc_gen.generate(descriptor, test_smiles)

        start = time.perf_counter()
        y_test_proba = ensemble.predict_proba()
        y_test_pred  = (y_test_proba >= 0.5).astype(int)
        pred_time = time.perf_counter() - start

        _save_metrics(test_smiles, y_test, y_test_pred, y_test_proba, pred_time, output_dir, "ENSEMBLE")

    if single_meta.exists():
        with open(single_meta) as f:
            meta = json.load(f)

        descriptor = meta['descriptor']
        model_name = meta['model']
        label = f"{descriptor} + {model_name}"

        model_file  = load_dir / meta['model_file']
        scaler_file = load_dir / meta['scaler_file']

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        logger.info(f"\nLoading {label} from {load_dir.name}/")
        clf    = joblib.load(model_file)
        scaler = joblib.load(scaler_file)

        desc_gen  = DescriptorGenerator(cache_dir=cache_dir)

        logger.info(f"Generating {descriptor} descriptors for test set...")
        X_test = desc_gen.generate(descriptor, test_smiles)
        X_test_scaled = scaler.transform(X_test) if scaler is not None else X_test

        start = time.perf_counter()
        y_test_pred  = clf.predict(X_test_scaled)
        pred_time = time.perf_counter() - start
        y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]

        _save_metrics(test_smiles, y_test, y_test_pred, y_test_proba, pred_time, output_dir, "BEST MODEL")

    if not (ensemble_meta.exists() or single_meta.exists()):
        raise FileNotFoundError(
            f"No model metadata found in {load_dir}. "
            "Expected model_metadata.json or ensemble_metadata.json."
        )

    end_time = datetime.now()
    logger.info(f"\nCOMPLETED  ({end_time - start_time})")
    logger.info(f"Output: {args.output}")


def _save_metrics(test_smiles, y_test, y_pred, y_proba, pred_time, output_dir, label):
    test_metrics = {
        **calculate_metrics(y_test, y_pred, y_proba),
        "Prediction Time": pred_time,
        "Prediction Time Normalized": pred_time / len(y_pred),
    }

    logger.info(f"\nTEST SET PERFORMANCE ({label})")
    for metric_name, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")

    pred_file    = output_dir / f'{label.lower().replace(" ", "_")}_test_predictions.csv'
    metrics_file = output_dir / f'{label.lower().replace(" ", "_")}_test_metrics.csv'

    pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Predicted_Probability': y_proba,
        'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_pred],
    }).to_csv(pred_file, index=False)

    pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Value':  list(test_metrics.values()),
    }).to_csv(metrics_file, index=False)


def _run_ensemble(args, output_dir, train_smiles, y_train,
                  test_smiles, y_test, start_time):
    """Run stacking ensemble pipeline."""

    top_k = select_top_k(args.results, k=args.top_k, metric=args.metric)

    logger.info(f"\nSELECTED TOP {args.top_k} MODELS (by {args.metric})")
    for i, (desc, mdl, score) in enumerate(top_k, 1):
        logger.info(f"  {i}. {desc} + {mdl} ({args.metric}={score:.4f})")

    base_configs = [(desc, mdl) for desc, mdl, _ in top_k]

    hyperparams = {}
    if args.hyperparams:
        hp_path = Path(args.hyperparams)
        if hp_path.exists():
            with open(hp_path) as f:
                hyperparams = json.load(f)
            logger.info(f"\nLoaded optimized hyperparameters from {hp_path}")
        else:
            logger.info(f"\nHyperparams file not found: {hp_path}, using defaults")

    cache_dir = output_dir / 'descriptor_cache'
    desc_gen = DescriptorGenerator(cache_dir=cache_dir)

    logger.info("\nBUILDING STACKING ENSEMBLE")
    ensemble_train_start = time.perf_counter()
    ensemble = StackingEnsemble(
        base_configs=base_configs,
        desc_generator=desc_gen,
        hyperparams=hyperparams,
        n_folds=Config.N_FOLDS,
        random_state=Config.RANDOM_STATE,
    )

    oof_predictions = ensemble.fit(train_smiles, y_train, test_smiles)
    ensemble_train_time = time.perf_counter() - ensemble_train_start
    with open(Path(output_dir) / 'ensemble_train_time.txt', "w") as file:
        file.write(str(ensemble_train_time))

    # OOF performance — use cross-validated meta-learner predictions to
    # avoid resubstitution bias (the meta-learner was trained on oof_predictions,
    # so evaluating it on the same data would be optimistically biased).
    meta_cv = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
    meta_oof_proba = cross_val_predict(
        LogisticRegression(max_iter=2000, random_state=Config.RANDOM_STATE),
        oof_predictions, y_train,
        cv=meta_cv,
        method='predict_proba',
    )[:, 1]
    oof_pred = (meta_oof_proba >= 0.5).astype(int)
    oof_metrics = calculate_metrics(y_train, oof_pred, meta_oof_proba)

    # Save the validation/oof metrics
    oof_metrics_df = pd.DataFrame({
        'Metric': list(oof_metrics.keys()),
        'Value': list(oof_metrics.values())
    })
    oof_metrics_file = output_dir / 'ensemble_oof_metrics.csv'
    oof_metrics_df.to_csv(oof_metrics_file, index=False)

    logger.info("\nENSEMBLE OOF PERFORMANCE (cross-validated meta-learner)")
    for metric_name, value in oof_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")

    logger.info("\nEVALUATING ENSEMBLE ON TEST SET")
    start = time.perf_counter()
    y_test_proba = ensemble.predict_proba()
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    ensemble_test_pred_time = time.perf_counter() - start
    test_metrics = {
        **calculate_metrics(y_test, y_test_pred, y_test_proba),
        "Prediction Time": ensemble_test_pred_time,
        "Prediction Time Normalized": ensemble_test_pred_time / len(y_test_pred),
    }

    logger.info("\nENSEMBLE TEST SET PERFORMANCE")
    for metric_name, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")

    logger.info("\nSAVING RESULTS")

    metadata = {
        'type': 'stacking_ensemble',
        'base_models': [
            {'descriptor': d, 'model': m} for d, m in base_configs
        ],
        'meta_learner': 'LogisticRegression',
        'top_k': args.top_k,
        'selection_metric': args.metric,
        'hyperparameters': {
            f"{d}_{m}": hyperparams.get(f"{d}_{m}", 'defaults')
            for d, m in base_configs
        },
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(y_train),
        'test_samples': len(y_test),
    }
    metadata_file = output_dir / 'ensemble_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata: {metadata_file.name}")

    pred_df = pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': y_test,
        'Predicted_Label': y_test_pred,
        'Predicted_Probability': y_test_proba,
        'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_test_pred]
    })
    pred_file = output_dir / 'ensemble_test_predictions.csv'
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"  Test predictions: {pred_file.name}")

    metrics_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Value': list(test_metrics.values())
    })
    metrics_file = output_dir / 'ensemble_test_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"  Test metrics: {metrics_file.name}")

    logger.info("\n" + "=" * 60)
    logger.info("BEST SINGLE MODEL (for comparison)")
    logger.info("=" * 60)

    best_descriptor, best_model = find_best_model(args.results, args.metric)

    # Reuse descriptors from the ensemble's cache
    X_train_best = ensemble.descriptors_train.get(best_descriptor)
    X_test_best = ensemble.descriptors_test.get(best_descriptor)

    if X_train_best is None:
        # Descriptor wasn't in the ensemble's top-K; generate it
        X_train_best, X_test_best = ensemble._generate_descriptors(
            best_descriptor, train_smiles, test_smiles
        )

    if best_descriptor.lower() == 'mordred':
        best_scaler = MinMaxScaler()
        X_train_best_scaled = best_scaler.fit_transform(X_train_best)
        X_test_best_scaled = best_scaler.transform(X_test_best)
    else:
        best_scaler = None
        X_train_best_scaled = X_train_best
        X_test_best_scaled = X_test_best

    best_params = hyperparams.get(f"{best_descriptor}_{best_model}", {})
    if best_params:
        logger.info(f"  Using optimized hyperparameters: {best_params}")

    clf = ModelFactory.create(best_model, **best_params)
    clf.fit(X_train_best_scaled, y_train)

    start = time.perf_counter()
    y_best_pred = clf.predict(X_test_best_scaled)
    best_model_test_pred_time = time.perf_counter() - start
    y_best_proba = clf.predict_proba(X_test_best_scaled)[:, 1]
    best_metrics = {
        **calculate_metrics(y_test, y_best_pred, y_best_proba),
        "Prediction Time": best_model_test_pred_time,
        "Prediction Time Normalized": best_model_test_pred_time / len(y_best_pred),
    }

    logger.info(f"\nBEST SINGLE MODEL TEST PERFORMANCE ({best_descriptor} + {best_model})")
    for metric_name, value in best_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")

    best_model_file = output_dir / f"{best_descriptor}_{best_model}_model.pkl"
    joblib.dump(clf, best_model_file)
    best_scaler_file = output_dir / f"{best_descriptor}_{best_model}_scaler.pkl"
    joblib.dump(best_scaler, best_scaler_file)

    metadata = {
        'descriptor': best_descriptor,
        'model': best_model,
        'scaler_type': 'MinMaxScaler' if best_descriptor.lower() == 'mordred' else 'None',
        'hyperparameters': best_params if best_params else 'defaults',
        'timestamp': datetime.now().isoformat(),
        'model_file': best_model_file.name,
        'scaler_file': best_scaler_file.name,
        'training_samples': len(y_train),
        'test_samples': len(y_test),
        'feature_dim': X_train_best.shape[1] if X_train_best.ndim > 1 else 'SMILES',
    }

    metadata_file = output_dir / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    best_pred_df = pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': y_test,
        'Predicted_Label': y_best_pred,
        'Predicted_Probability': y_best_proba,
        'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_best_pred]
    })
    best_pred_file = output_dir / 'best_model_test_predictions.csv'
    best_pred_df.to_csv(best_pred_file, index=False)

    best_metrics_df = pd.DataFrame({
        'Metric': list(best_metrics.keys()),
        'Value': list(best_metrics.values())
    })
    best_metrics_file = output_dir / 'best_model_test_metrics.csv'
    best_metrics_df.to_csv(best_metrics_file, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: ENSEMBLE vs BEST SINGLE MODEL")
    logger.info("=" * 60)
    logger.info(f"  {'Metric':<20} {'Ensemble':>12} {best_descriptor + ' + ' + best_model:>25} {'Diff':>10}")
    logger.info(f"  {'-'*20} {'-'*12} {'-'*25} {'-'*10}")

    for metric_name in test_metrics:
        ens_val = test_metrics[metric_name]
        best_val = best_metrics[metric_name]
        if isinstance(ens_val, float) and isinstance(best_val, float):
            diff = ens_val - best_val
            sign = '+' if diff >= 0 else ''
            logger.info(f"  {metric_name:<20} {ens_val:>12.4f} {best_val:>25.4f} {sign}{diff:>9.4f}")

    comparison_rows = []
    for metric_name in test_metrics:
        ens_val = test_metrics[metric_name]
        best_val = best_metrics[metric_name]
        if isinstance(ens_val, float) and isinstance(best_val, float):
            comparison_rows.append({
                'Metric': metric_name,
                'Ensemble': ens_val,
                f'{best_descriptor}_{best_model}': best_val,
                'Difference': ens_val - best_val,
            })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_file = output_dir / 'ensemble_vs_best_model_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\n  Comparison saved: {comparison_file.name}")

    # Save ensemble model (clear large cached arrays but keep desc_gen for
    # predict_proba_new() on future datasets)
    ensemble_file = output_dir / f'stacking_ensemble_top{args.top_k}.pkl'
    ensemble.descriptors_train = {}
    ensemble.descriptors_test = {}
    joblib.dump(ensemble, ensemble_file)
    logger.info(f"  Ensemble: {ensemble_file.name}")

    end_time = datetime.now()
    logger.info(f"\nCOMPLETED  ({end_time - start_time})")
    logger.info(f"Output: {args.output}")


def _run_single_model(args, output_dir, train_smiles, y_train,
                      test_smiles, y_test, start_time):
    """Train a single descriptor–model combination on all training data and evaluate on test."""

    if args.results:
        descriptor, model = find_best_model(args.results, args.metric)
    else:
        descriptor = args.descriptor
        model = args.model
        logger.info("\nMODEL SELECTION")
        logger.info(f"  Descriptor: {descriptor}")
        logger.info(f"  Model: {model}")

    cache_dir = output_dir / 'descriptor_cache'
    desc_gen = DescriptorGenerator(cache_dir=cache_dir)

    all_smiles = train_smiles + test_smiles
    X_all = desc_gen.generate(descriptor, all_smiles)

    n_train = len(train_smiles)
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    logger.info("\nTRAINING MODEL ON ALL TRAINING DATA")

    if descriptor.lower() == 'mordred':
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test

    model_params = {}
    if args.hyperparams:
        hp_path = Path(args.hyperparams)
        if hp_path.exists():
            with open(hp_path) as f:
                all_params = json.load(f)
            key = f"{descriptor}_{model}"
            model_params = all_params.get(key, {})
            if model_params:
                logger.info(f"\nUsing optimized hyperparameters: {model_params}")
            else:
                logger.info(f"\nNo optimized params found for {key}, using defaults")
        else:
            logger.info(f"\nHyperparams file not found: {hp_path}, using defaults")

    clf = ModelFactory.create(model, **model_params)
    clf.fit(X_train_scaled, y_train)

    y_train_pred = clf.predict(X_train_scaled)
    y_train_proba = clf.predict_proba(X_train_scaled)[:, 1]
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)

    logger.info("\nTRAINING SET PERFORMANCE")
    for metric, value in train_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")

    y_test_pred = clf.predict(X_test_scaled)
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    logger.info("\nTEST SET PERFORMANCE")
    for metric, value in test_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")

    model_file = output_dir / f"{descriptor}_{model}_model.pkl"
    joblib.dump(clf, model_file)

    scaler_file = output_dir / f"{descriptor}_{model}_scaler.pkl"
    joblib.dump(scaler, scaler_file)

    metadata = {
        'descriptor': descriptor,
        'model': model,
        'scaler_type': 'MinMaxScaler' if descriptor.lower() == 'mordred' else 'None',
        'hyperparameters': model_params if model_params else 'defaults',
        'timestamp': datetime.now().isoformat(),
        'model_file': model_file.name,
        'scaler_file': scaler_file.name,
        'training_samples': len(y_train),
        'test_samples': len(y_test),
        'feature_dim': X_train.shape[1] if X_train.ndim > 1 else 'SMILES',
    }

    metadata_file = output_dir / 'model_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    pred_df = pd.DataFrame({
        'SMILES': test_smiles,
        'True_Label': y_test,
        'Predicted_Label': y_test_pred,
        'Predicted_Probability': y_test_proba,
        'Prediction': ['Toxic' if p == 1 else 'Non-toxic' for p in y_test_pred]
    })
    pred_file = output_dir / 'test_predictions.csv'
    pred_df.to_csv(pred_file, index=False)

    metrics_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Value': list(test_metrics.values())
    })
    metrics_file = output_dir / 'test_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)

    end_time = datetime.now()
    logger.info(f"\nCOMPLETED  ({end_time - start_time})")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
