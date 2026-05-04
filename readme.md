# AutoML

This repository contains code to reproduce exploration and comparison of AutoML frameworks as applied to 5 binary classification ADMET endpoints: Chemical Mutagenicity (AMES), Drug-Induced Liver Injury (DILI), Cytotoxicity (Cytotox), Human Liver Microsomal Stability (HLM), and Mitochondrial Membrane Potential Disruption (MMP).

All scripts in this repository are meant to be run in the base directory of the repository, i.e. the directory this readme is in.

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

# Preprocessing

To preprocess the raw data, run the following:
```
.venv-preprocessing\Scripts\python.exe preprocessing\preprocess_all_5_endpoints.py
```
This will generate 2048-bit Morgan fingerprints and Mordred descriptors for the compounds in all 5 datasets and separate them into a train-test scaffold split. The preprocessed data will appear under `data/preprocessed`.

# AutoML

To fit predictors using the frameworks run the following:

**Activate the virtual environment:** `.venv-automl\Scripts\Activate.ps1`

**vNN:** `vnn\vnn\vnn_all_endpoints_hpo.py`

**AutoGluon:** `autogluon\autogluon_all_endpoints_training.py`

**FLAML:** `flaml\flaml_all_endpoints_training.py`

All three of the above scripts include an option `--use_prefit` to skip fitting a predictor and use a previously fitted predictor to generate performance metrics—useful if you are changing how the performance metrics are reported.

# Visualization

Run the following to generate visualizations:

`python visualization\plot_performance.py`
