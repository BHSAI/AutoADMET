# Developing machine learning models to predict chemical-breast cancer resistance protein (BCRP) interaction

A machine learning pipeline for binary toxicity prediction with statistical analysis and visualizations. Adaptable to any endpoint.

## Features

- **7 Molecular Descriptors**: Morgan, MACCS, RDKit, Mordred, ChemBERTa, MolFormer, SMILES (pass-through for GNNs)
- **10 ML Models**: KNN, SVM, Bayesian, Logistic Regression, Random Forest, LightGBM, XGBoost, TabPFN, Chemprop, CheMeleon
- **Hyperparameter Optimization**: Optuna-based tuning per model-descriptor pair
- **Stacking Ensemble**: Combine top-K model-descriptor pairs via stacking with a meta-learner
- **Rigorous Cross-Validation**: 5-repeat x 5-fold stratified CV (configurable)
- **Descriptor Caching**: Automatic caching of computed descriptors for faster re-runs

## Installation

**Prerequisites:** Python 3.11+, CUDA (optional, for GPU acceleration)

```bash
git clone https://github.com/BHSAI/bcrp_classify.git
cd bcrp_classify
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For TabPFN, accept the model license and configure the access token in Hugging Face during first use ([details](https://docs.priorlabs.ai/how-to-access-gated-models)).

## Quick Start

### Step 1: Data Preprocessing

```bash
python preprocess_data.py --input data/BCRP_inhibitors_Jiang_2020.csv --output-dir data/

# Plot-only mode (if train/test CSVs already exist)
python preprocess_data.py --plot-only data/train_df.csv data/test_df.csv -o data/
```

Validates SMILES, standardizes structures, removes duplicates/conflicts, performs Butina clustering, creates train/test split, and analyzes similarity.

### Step 2: Cross-Validation Pipeline

```bash
python pipeline.py --input data/train_df.csv --output cv_results/
```

Tests all descriptor-model combinations with repeated stratified CV, optional Optuna optimization, and generates statistical tests and visualizations.

```bash
python pipeline.py --input data/train_df.csv --output cv_results/ \
    --hyperparams cv_results/optimized_hyperparameters.json
```

Skips Optuna and loads hyperparameters from an existing JSON file. Useful when re-running CV after a prior optimization run.

### Step 3: Train Final Model

```bash
python train_test_best_model.py \
    --train data/train_df.csv --test data/test_df.csv \
    --results cv_results/ \
    --hyperparams cv_results/optimized_hyperparameters.json \
    --output final_model/
```

Auto-selects the best model from CV results. The `--hyperparams` flag is optional (uses defaults when omitted). External validation datasets can be passed via `--test` in place of or in addition to the internal test set.

To evaluate a previously saved model on a new test set without retraining:

```bash
python train_test_best_model.py \
    --load-model final_model/ \
    --test external_test.csv \
    --output external_results/
```

### Step 4: Stacking Ensemble

```bash
python train_test_best_model.py \
    --train data/train_df.csv --test data/test_df.csv \
    --results cv_results/ --ensemble --top-k 5 \
    --hyperparams cv_results/optimized_hyperparameters.json \
    --output ensemble_model/
```

Combines top-K model-descriptor pairs via stacking. Also trains the single best model for comparison.

## Configuration

All pipeline settings are in the `Config` class in `config.py`:

```python
class Config:
    N_REPEATS = 5
    N_FOLDS = 5
    RANDOM_STATE = 42
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 2048
    ENABLE_CACHE = True
    CACHE_DIR = 'descriptor_cache'

    DESCRIPTORS = ['Morgan', 'RDKit', 'MACCS', 'Mordred', 'ChemBERTa', 'MolFormer', 'SMILES']
    MODELS = ['KNN', 'SVM', 'Bayesian', 'LogisticRegression', 'RandomForest',
              'LightGBM', 'XGBoost', 'TabPFN', 'Chemprop', 'CheMeleon']
    METRICS = ['ROC_AUC', 'PR_AUC', 'Accuracy', 'Sensitivity', 'Specificity',
               'GMean', 'Precision', 'F1', 'MCC', 'Kappa']

    ENABLE_OPTUNA = True
    OPTUNA_N_TRIALS = 25
    OPTUNA_METRIC = 'MCC'
    STATS_METRICS = ['MCC', 'ROC_AUC', 'GMean']
    VIZ_METRICS = ['MCC', 'ROC_AUC', 'GMean']
    VIZ_DPI = 300
```

## Descriptor-Model Compatibility

| | Traditional ML | GNN Models (Chemprop, CheMeleon) |
|---|---|---|
| **Feature descriptors** (Morgan, RDKit, MACCS, Mordred, ChemBERTa, MolFormer) | Compatible | Skipped |
| **SMILES** (pass-through) | Skipped | Compatible |

## Data

The `data/` directory contains the datasets used in this study:

- **`train_df.csv` / `test_df.csv`**: Training set and cluster-split internal test set, both derived from [Jiang et al. 2020](https://link.springer.com/article/10.1186/s13321-020-00421-y) via Butina clustering.
- **`BCRP_inhibitors_Jiang_2020.csv`**: Full source dataset from Jiang et al. 2020 (pre-split).
- **`BCRP_inhibitors_Daood_2025.csv`**: Independent external test set from [Daood et al. 2025](https://pubs.acs.org/doi/full/10.1021/acs.molpharmaceut.5c01065), used for external validation via `train_test_best_model.py`.

To use your own data, provide a CSV with `SMILES` and `CLASS` columns and run Step 1.

## Input Format

CSV with columns `SMILES` and `CLASS` (binary: 0/1). Invalid SMILES are filtered automatically.

```csv
SMILES,CLASS
C[C@]12CC[C@@H]3c4ccc(OS(=O)(=O)O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O,1
CNc1nc2c(Cc3cccnc3)c(C)c(OS(=O)(=O)O)c(C)c2s1,1
O=C1c2c(O)ccc(O)c2C(=O)c2c(NCCNCCO)ccc(NCCNCCO)c21,0
CC[C@H](C)C(=O)O[C@H]1C[C@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@H]21,0
```

## Methodology

**Descriptors:** Morgan (ECFP4), MACCS (166-bit), RDKit, Mordred (1613 2D/3D), ChemBERTa, MolFormer, SMILES pass-through.

**Models:** KNN, SVM, Bayesian (BernoulliNB), Logistic Regression, Random Forest, LightGBM, XGBoost, TabPFN, Chemprop (D-MPNN), CheMeleon (pretrained D-MPNN). Hyperparameters for all traditional ML models can be tuned via Optuna.

**Hyperparameter Optimization:** Optuna uses Tree-structured Parzen Estimators (TPE), a sequential model-based Bayesian optimization algorithm. TPE fits two non-parametric density estimates over the hyperparameter space — one for configurations that achieved good objective values and one for the rest — and samples candidates that maximize the ratio between the two. Each trial's result updates both distributions, so later trials are increasingly directed toward promising regions. Optimization is performed independently per descriptor–model pair using the MCC score (configurable) averaged over the inner CV folds as the objective. A total of 25 trials are run per pair by default (configurable via `OPTUNA_N_TRIALS`).

**Cross-Validation:** Repeated stratified K-fold with consistent seeds.

**Statistical Analysis:** RM-ANOVA, Tukey HSD post-hoc, and Critical Difference diagrams (Conover-Friedman).

**Stacking Ensemble:** OOF predictions from top-K base models feed a Logistic Regression meta-learner, preventing information leakage.
