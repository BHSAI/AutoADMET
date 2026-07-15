"""Main pipeline: descriptor generation, cross-validation, statistical analysis, and visualization."""

import os
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''

import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd

# Suppress common warnings
warnings.filterwarnings('ignore', message='PYTORCH_CUDA_ALLOC_CONF is deprecated')

# Lightweight imports (always needed)
from config import Config
from utils.stats import StatisticalAnalyzer, plot_combined_cd, prepare_subject_column
from utils.plots import Visualizer

import logging

logger = logging.getLogger(__name__)

# Heavy imports — deferred so --plot-only works without full dependencies
_HEAVY_IMPORTS_LOADED = False

def _load_heavy_imports():
    """Import modules that require rdkit, mordred, pytorch, etc."""
    global _HEAVY_IMPORTS_LOADED
    if _HEAVY_IMPORTS_LOADED:
        return
    global tqdm, Chem
    global DescriptorGenerator, get_available_descriptors
    global CrossValidator, get_available_models, SMILES_MODELS
    global HyperparameterOptimizer, save_hyperparameters

    from tqdm import tqdm as _tqdm
    from rdkit import Chem as _Chem
    from utils.descriptors import DescriptorGenerator as _DG, get_available_descriptors as _gad
    from utils.models import CrossValidator as _CV, get_available_models as _gam, SMILES_MODELS as _SM
    from utils.optimization import HyperparameterOptimizer as _HO, save_hyperparameters as _shp

    tqdm = _tqdm
    Chem = _Chem
    DescriptorGenerator = _DG
    get_available_descriptors = _gad
    CrossValidator = _CV
    get_available_models = _gam
    SMILES_MODELS = _SM
    HyperparameterOptimizer = _HO
    save_hyperparameters = _shp
    _HEAVY_IMPORTS_LOADED = True


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class ToxicityPipeline:
    """
    Main pipeline orchestrator.
    
    Coordinates data loading, descriptor generation, model training,
    statistical analysis, and visualization.
    """
    
    def __init__(self, input_file, output_dir, hyperparams_file=None):
        """
        Initialize pipeline.

        Args:
            input_file: Path to CSV with SMILES and CLASS columns
            output_dir: Directory for results
            hyperparams_file: Optional path to pre-computed hyperparameters JSON
                              (skips Optuna when provided)
        """
        _load_heavy_imports()

        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hyperparams_file = Path(hyperparams_file) if hyperparams_file else None

        self.plots_dir = self.output_dir / 'plots'
        self.stats_dir = self.output_dir / 'statistical_tests'
        self.plots_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)

        cache_dir = self.output_dir / Config.CACHE_DIR if Config.ENABLE_CACHE else None

        self.desc_gen = DescriptorGenerator(
            morgan_radius=Config.MORGAN_RADIUS,
            morgan_nbits=Config.MORGAN_NBITS,
            cache_dir=cache_dir
        )
        
        self.cv = CrossValidator(
            n_repeats=Config.N_REPEATS,
            n_folds=Config.N_FOLDS,
            random_state=Config.RANDOM_STATE
        )
        
        self.stats = StatisticalAnalyzer()
        self.viz = Visualizer(self.plots_dir, dpi=Config.VIZ_DPI)
        
        logger.info(f"\nPipeline initialized")
        logger.info(f"Input: {input_file}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Device: {self.desc_gen.device}\n")
    
    def load_data(self):
        """
        Load and validate data from CSV.
        
        Returns:
            Tuple of (smiles_list, labels)
        """
        logger.info("STEP 1: Loading Data")

        df = pd.read_csv(self.input_file)

        if 'SMILES' not in df.columns or 'CLASS' not in df.columns:
            raise ValueError("CSV must have 'SMILES' and 'CLASS' columns")

        df = df.dropna(subset=['SMILES', 'CLASS']).reset_index(drop=True)

        logger.info("Validating SMILES...")
        valid_smiles = []
        valid_labels = []
        
        for smi, label in tqdm(zip(df['SMILES'], df['CLASS']), total=len(df), desc="  Validating SMILES", leave=False):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_labels.append(label)

        logger.info(f"  Total molecules: {len(df)}")
        logger.info(f"  Valid molecules: {len(valid_smiles)}")
        logger.info(f"  Invalid SMILES: {len(df) - len(valid_smiles)}")
        logger.info(f"  Class distribution: {np.bincount(valid_labels)}")
        
        return valid_smiles, np.array(valid_labels)
    
    def generate_descriptors(self, smiles_list):
        """
        Generate all configured descriptors.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary mapping descriptor_name -> scaled_features
        """
        logger.info("\nSTEP 2: Generating Descriptors")
        
        available = get_available_descriptors()
        descriptors = {}
        
        for desc_type in Config.DESCRIPTORS:
            if desc_type not in available:
                logger.info(f"\nSkipping {desc_type} (not available)")
                continue
            
            try:
                logger.info(f"\n{desc_type}:")
                X = self.desc_gen.generate(desc_type, smiles_list)
                if desc_type == 'SMILES':
                    logger.info(f"  Pass-through: {len(X)} SMILES for GNN models")
                else:
                    logger.info(f"  Shape: {X.shape}")
                
                descriptors[desc_type] = X
                
            except Exception as e:
                logger.info(f"  Error: {e}")
        
        return descriptors
    
    def optimize_hyperparameters(self, descriptors, y):
        """
        Run Optuna hyperparameter optimization for all model-descriptor pairs.

        Args:
            descriptors: Dictionary of descriptor_name -> features
            y: Labels

        Returns:
            Dictionary mapping "descriptor_model" -> best_params
        """
        logger.info("\nSTEP 3: Hyperparameter Optimization (Optuna)")
        logger.info(f"Trials per combination: {Config.OPTUNA_N_TRIALS}")
        logger.info(f"Optimizing: {Config.OPTUNA_METRIC}")

        optimizer = HyperparameterOptimizer(
            n_repeats=Config.N_REPEATS,
            n_folds=Config.N_FOLDS,
            random_state=Config.RANDOM_STATE,
            device=self.cv.device,
        )

        available_models = get_available_models()
        active_models = [m for m in Config.MODELS if m in available_models]

        best_params = optimizer.optimize_all(
            descriptors, y, active_models,
            n_trials=Config.OPTUNA_N_TRIALS,
            metric=Config.OPTUNA_METRIC,
        )

        params_file = self.output_dir / 'optimized_hyperparameters.json'
        save_hyperparameters(best_params, params_file)

        return best_params

    def train_models(self, descriptors, y, optimized_params=None):
        """
        Train all models on all descriptors with cross-validation.

        Args:
            descriptors: Dictionary of descriptor_name -> features
            y: Labels
            optimized_params: Optional dict from optimize_hyperparameters()

        Returns:
            DataFrame with per-fold results
        """
        step = "STEP 4" if Config.ENABLE_OPTUNA else "STEP 3"
        logger.info(f"\n{step}: Training Models (CV)")

        if optimized_params is None:
            optimized_params = {}

        available_models = get_available_models()
        all_results = []

        for desc_name, X in descriptors.items():
            logger.info(f"\n{desc_name}:")

            for model_name in Config.MODELS:
                if model_name not in available_models:
                    logger.info(f"  Skipping {model_name} (not available)")
                    continue

                if desc_name == 'SMILES' and model_name not in SMILES_MODELS:
                    continue
                if desc_name != 'SMILES' and model_name in SMILES_MODELS:
                    continue

                try:
                    key = f"{desc_name}_{model_name}"
                    params = optimized_params.get(key, {})

                    if params:
                        logger.info(f"  Training {model_name} (optimized)...")
                    else:
                        logger.info(f"  Training {model_name}...")

                    results = self.cv.run_cv(
                        X, y, model_name, desc_name, model_params=params
                    )
                    all_results.extend(results)

                except Exception as e:
                    logger.info(f"    Error: {e}")

        results_df = pd.DataFrame(all_results)

        results_file = self.output_dir / 'per_fold_results.csv'
        results_df.to_csv(results_file, index=False)
        logger.info(f"\nSaved per-fold results: {results_file}")

        return results_df
    
    def create_summary(self, results_df):
        """
        Create summary table with mean metrics.
        
        Args:
            results_df: DataFrame with per-fold results
            
        Returns:
            DataFrame with summary statistics
        """
        step = "STEP 5" if Config.ENABLE_OPTUNA else "STEP 4"
        logger.info(f"\n{step}: Creating Summary")

        summary = results_df.groupby(['Descriptor', 'Model'])[Config.METRICS].mean().reset_index()
        for metric in Config.METRICS:
            summary[metric] = summary[metric].round(3)

        summary_file = self.output_dir / 'results_summary.csv'
        summary.to_csv(summary_file, index=False)
        logger.info(f"Saved summary: {summary_file}")

        best_idx = summary['MCC'].idxmax()
        best = summary.iloc[best_idx]

        logger.info(f"\nBest model (by MCC):")
        logger.info(f"  Descriptor: {best['Descriptor']}")
        logger.info(f"  Model: {best['Model']}")
        logger.info(f"  MCC: {best['MCC']:.3f}")
        logger.info(f"  ROC-AUC: {best['ROC_AUC']:.3f}")
        logger.info(f"  GMean: {best['GMean']:.3f}")
        
        return summary
    
    def run_statistical_analysis(self, results_df):
        """
        Perform statistical tests.
        
        Args:
            results_df: DataFrame with per-fold results
        """
        step = "STEP 6" if Config.ENABLE_OPTUNA else "STEP 5"
        logger.info(f"\n{step}: Statistical Analysis")
        
        for descriptor in results_df['Descriptor'].unique():
            logger.info(f"\n{descriptor}:")
            
            for metric in Config.STATS_METRICS:
                anova_res = self.stats.run_anova(results_df, descriptor, metric)
                if anova_res is not None:
                    p_value = anova_res.anova_table['Pr > F'].iloc[0]
                    f_value = anova_res.anova_table['F Value'].iloc[0]
                    logger.info(f"  {metric}: F={f_value:.2f}, p-value={p_value:.4f}")

                tukey_res = self.stats.tukey_hsd(results_df, descriptor, metric)
                tukey_file = self.stats_dir / f"{descriptor}_{metric}_Tukey.csv"
                tukey_res.to_csv(tukey_file, index=False)
    
    def create_visualizations(self, results_df, summary_df):
        """
        Create all visualizations.
        
        Args:
            results_df: Per-fold results
            summary_df: Summary statistics
        """
        step = "STEP 7" if Config.ENABLE_OPTUNA else "STEP 6"
        logger.info(f"\n{step}: Creating Visualizations")

        for metric in Config.VIZ_METRICS:
            filename = self.viz.plot_heatmap(
                summary_df, metric,
                descriptor_order=Config.DESCRIPTORS,
                model_order=Config.MODELS
            )
            logger.info(f"  Saved: {filename.name}")

        filename = self.viz.plot_comparison(summary_df, metrics=Config.VIZ_METRICS,
                                            descriptor_order=Config.DESCRIPTORS,
                                            model_order=Config.MODELS)
        logger.info(f"  Saved: {filename.name}")

        for metric in Config.VIZ_METRICS:
            filename = self.viz.create_combined_boxplot(
                results_df, metric,
                descriptor_order=Config.DESCRIPTORS,
                model_order=Config.MODELS
            )
            logger.info(f"  Saved: {filename.name}")

        from utils.plots import plot_top10_boxplot
        bp_path = self.plots_dir / "top10_models_boxplot.png"
        plot_top10_boxplot(results_df, bp_path)
        logger.info(f"  Saved: {bp_path.name}")

        df_cd = prepare_subject_column(results_df.copy())
        df_cd["Combo"] = df_cd["Descriptor"] + " + " + df_cd["Model"]
        top10 = (
            df_cd.groupby("Combo")[Config.OPTUNA_METRIC].mean()
            .sort_values(ascending=False)
            .head(10).index.tolist()
        )
        df_top = df_cd[df_cd["Combo"].isin(top10)]
        cd_path = self.plots_dir / "cd_combined.png"
        plot_combined_cd(df_top, {"Kappa": "Kappa"}, str(cd_path))
        logger.info(f"  Saved: {cd_path.name}")

        logger.info(f"\nPlots saved to: {self.plots_dir}")
    
    def run(self):
        """
        Run the complete pipeline.
        """
        start_time = datetime.now()
        
        logger.info("\nTOXICITY CLASSIFICATION PIPELINE")
        logger.info(f"Started: {start_time:%Y-%m-%d %H:%M:%S}\n")
        logger.info(f"Descriptors: {', '.join(Config.DESCRIPTORS)}")
        logger.info(f"Models: {', '.join(Config.MODELS)}")
        logger.info(f"Cross-validation: {Config.N_REPEATS}×{Config.N_FOLDS}")
        if Config.ENABLE_OPTUNA:
            logger.info(f"Optuna: {Config.OPTUNA_N_TRIALS} trials, optimizing {Config.OPTUNA_METRIC}")

        try:
            smiles_list, y = self.load_data()
            descriptors = self.generate_descriptors(smiles_list)

            train_start = time.perf_counter()
            optimized_params = {}
            if self.hyperparams_file is not None:
                with open(self.hyperparams_file) as f:
                    optimized_params = json.load(f)
                logger.info(f"\nLoaded hyperparameters from {self.hyperparams_file}")
            elif Config.ENABLE_OPTUNA:
                optimized_params = self.optimize_hyperparameters(descriptors, y)

            results_df = self.train_models(descriptors, y, optimized_params)
            train_time = time.perf_counter() - train_start
            with open(self.output_dir / 'train_time.txt', "w") as file:
                file.write(str(train_time))
            summary_df = self.create_summary(results_df)
            self.run_statistical_analysis(results_df)
            self.create_visualizations(results_df, summary_df)

            end_time = datetime.now()
            logger.info(f"\nPIPELINE COMPLETED  ({end_time - start_time})")
            logger.info(f"Results: {self.output_dir}")

        except Exception as e:
            logger.info(f"\nERROR: {e}")
            raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Toxicity Classification Pipeline",
    )
    
    parser.add_argument(
        '--input', type=str, default=None,
        help='Input CSV file with SMILES and CLASS columns'
    )
    
    parser.add_argument(
        '--output', type=str, default='results',
        help='Output directory (default: results)'
    )

    parser.add_argument(
        '--hyperparams', type=str, default=None,
        help='Path to pre-computed optimized_hyperparameters.json (skips Optuna)'
    )

    parser.add_argument(
        '--plot-only', action='store_true',
        help='Regenerate plots from existing per_fold_results.csv and '
             'results_summary.csv in the output directory (skips training)'
    )

    args = parser.parse_args()

    if not args.plot_only and args.input is None:
        parser.error("--input is required when not using --plot-only")

    if args.plot_only:
        output_dir = Path(args.output)
        per_fold_file = output_dir / 'per_fold_results.csv'
        summary_file = output_dir / 'results_summary.csv'

        for f in (per_fold_file, summary_file):
            if not f.exists():
                parser.error(f"--plot-only requires {f} (run the full pipeline first)")

        results_df = pd.read_csv(per_fold_file)
        summary_df = pd.read_csv(summary_file)

        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        viz = Visualizer(plots_dir, dpi=Config.VIZ_DPI)
        stats = StatisticalAnalyzer()

        logger.info("\nPLOT-ONLY MODE")
        logger.info(f"  Per-fold results: {per_fold_file} ({len(results_df)} rows)")
        logger.info(f"  Summary:          {summary_file} ({len(summary_df)} rows)")

        for metric in Config.VIZ_METRICS:
            filename = viz.plot_heatmap(
                summary_df, metric,
                descriptor_order=Config.DESCRIPTORS,
                model_order=Config.MODELS
            )
            logger.info(f"  Saved: {filename.name}")

        filename = viz.plot_comparison(summary_df, metrics=Config.VIZ_METRICS, descriptor_order=Config.DESCRIPTORS, model_order=Config.MODELS)
        logger.info(f"  Saved: {filename.name}")

        pipeline = ToxicityPipeline(None, args.output, None)
        pipeline.create_visualizations(results_df, summary_df)

        for metric in Config.VIZ_METRICS:
            filename = viz.create_combined_boxplot(
                results_df, metric,
                descriptor_order=Config.DESCRIPTORS,
                model_order=Config.MODELS
            )
            logger.info(f"  Saved: {filename.name}")

        ext_csv = output_dir / 'ext_performance_compare.csv'
        if not ext_csv.exists():
            ext_csv = Path('ext_performance_compare.csv')
        if ext_csv.exists():
            from utils.plots import plot_ext_performance_comparison
            saved = plot_ext_performance_comparison(ext_csv, save_path=plots_dir / "ext_performance_comparison.png")
            logger.info(f"  Saved: {Path(saved).name}")
        else:
            logger.info("\nSkipping external performance comparison (ext_performance_compare.csv not found)")

        logger.info(f"\nPlots: {plots_dir}")
    else:
        pipeline = ToxicityPipeline(args.input, args.output, args.hyperparams)
        pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
