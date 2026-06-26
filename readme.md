# AutoML

This repository contains code to reproduce comparison of BHSAI's AutoML pipeline to the vNN-ADMET algorithm as applied to 14 binary classification ADMET endpoints: 
- Chemical Mutagenicity (AMES)
- Blood-Brain Barrier (BBB)
- Cytochrome P450 Inhibition 1A2 (CYP1A2)
- Cytochrome P450 Inhibition 2C9 (CYP2C9)
- Cytochrome P450 Inhibition 2C19 (CYP2C19)
- Cytochrome P450 Inhibition 2D6 (CYP2D6)
- Cytochrome P450 Inhibition 3A4 (CYP3A4)
- Cytotoxicity (Cytotox)
- Drug-Induced Liver Injury (DILI)
- Cardiotoxicitiy (hERG)
- Human Liver Microsomal Stability (HLM)
- Mitochondrial Membrane Potential Disruption (MMP)
- Pgp Inhibitors
- Pgp Substrates

All scripts in this repository are meant to be run from the base directory of the repository, i.e. the directory this readme is in.

# Setup

This code has been tested with Python 3.12 (specifically, 3.12.9). As such we recommend using an installation of Python 3.12 to run it. To set up the virtual environments run the following commands from this directory:

#### Windows
Replace `[python-3.12]` with the location of your Python 3.12 executable, e.g. `~\AppData\Local\Programs\Python\Python312\python.exe`.
```
[python-3.12] -m venv .venv-preprocessing
.venv-preprocessing\Scripts\pip.exe install -r requirements-preprocessing.txt

[python-3.12] -m venv .venv-automl
.venv-automl\Scripts\pip.exe install -r requirements-automl.txt
```

#### Linux
```
[python-3.12] -m venv .venv-preprocessing
.venv-preprocessing\bin\pip install -r requirements-preprocessing.txt

[python-3.12] -m venv .venv-automl
.venv-automl\bin\pip install -r requirements-automl.txt
```

# Preprocessing

To preprocess the raw data, run the following:
```
.venv-preprocessing\Scripts\python.exe preprocessing\preprocess_all_5_endpoints.py
```
This will generate 2048-bit Morgan fingerprints and Mordred descriptors for the compounds in all 5 datasets and separate them into a train-test scaffold split. The preprocessed data will appear under `data/preprocessed`.

# AutoML

To fit predictors using the frameworks run the following:

**Activate the virtual environment:** `.venv-automl\Scripts\Activate.ps1`

**vNN:** `python vnn\vnn\vnn_all_endpoints_hpo.py`

**BHSAI AutoML Pipeline:** `python bhsai_automl_pipeline/bhsai_pipeline_all_endpoints_training.py`

Both of the above scripts include options to skip the time-consuming hyperparameter search. For vNN, use `--use_prefit`. For BHSAI AutoML Pipeline, use `--mode` with option `skip-cv` to skip the hyperparameter search, `load-model` to skip the ensembling and refitting the optimal model with full data, or `parse-results-only` to skip model evaluation and regenerate performance plots with previously collected performance data.

# Visualization

Run the following to generate validation performance comparisons between vNN and the individual models that the BHSAI AutoML Pipeline trains:
`python visualization\plot_significance_tests.py`

And the following to generate evaluation performance comparisons between vNN, the best validating model from the BHSAI AutoML Pipeline, and its ensemble model:
`python visualization\plot_performance.py`
