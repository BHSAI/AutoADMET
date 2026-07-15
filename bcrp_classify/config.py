"""All pipeline settings in one place. Edit this file to customize descriptors, models, CV, and Optuna options."""


class Config:

    # Cross-validation settings
    N_REPEATS = 5
    N_FOLDS = 5
    RANDOM_STATE = 42

    # Descriptor settings
    MORGAN_RADIUS = 2
    MORGAN_NBITS = 2048

    # Descriptor caching
    ENABLE_CACHE = True  # Set to False to disable caching
    CACHE_DIR = 'descriptor_cache'  # Directory to store cached descriptors

    # Which descriptors to use (comment out to skip)
    # Note: 'SMILES' is a pass-through for GNN models (Chemprop, CheMeleon)
    DESCRIPTORS = [
        'Morgan',
        'Mordred',
    ]

    # Which models to use (comment out to skip)
    # Note: Chemprop/CheMeleon are GNNs that operate on SMILES directly
    MODELS = [
        # Comment out vNN for training step, only include for plot generation
        "vNN",
        'KNN',
        'Bayesian',
        'LogisticRegression',
        'RandomForest',
        'LightGBM',
        'XGBoost',
    ]

    # Metrics to calculate
    METRICS = [
        'ROC_AUC',
        'PR_AUC',
        'Accuracy',
        'Sensitivity',
        'Specificity',
        'GMean',
        'Precision',
        'F1',
        'MCC',
        'Kappa'
    ]

    # Statistical tests settings
    STATS_METRICS = ['MCC', 'ROC_AUC', 'GMean', 'Kappa']  # Metrics to run stats on

    # Visualization settings
    VIZ_METRICS = ['MCC', 'ROC_AUC', 'GMean', 'Kappa']  # Metrics to plot
    VIZ_DPI = 300

    # Hyperparameter optimization (Optuna)
    ENABLE_OPTUNA = True   # Set to False to skip optimization
    OPTUNA_N_TRIALS = 25   # Number of Optuna trials per model-descriptor pair
    OPTUNA_METRIC = 'Kappa'  # Metric to optimize
