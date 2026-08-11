# AutoADMET

This repository contains code to reproduce comparison of AutoADMET pipeline to the vNN-ADMET algorithm as applied to 14 binary classification ADMET endpoints: 
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

Replace `[python-3.12]` with the location of your Python 3.12 executable, e.g. `~\AppData\Local\Programs\Python\Python312\python.exe`.

``` Powershell
# Windows
[python-3.12] -m venv .venv-preprocessing
.venv-preprocessing\Scripts\pip.exe install -r requirements-preprocessing.txt

[python-3.12] -m venv .venv-automl
.venv-automl\Scripts\pip.exe install -r requirements-automl.txt

[python-3.12] -m venv bcrp_classify\.venv
bcrp_classify\.venv\Scripts\pip.exe install -r bcrp_classify\requirements.txt
```

``` sh
# Linux
[python-3.12] -m venv .venv-preprocessing
.venv-preprocessing/bin/pip install -r requirements-preprocessing.txt

[python-3.12] -m venv .venv-automl
.venv-automl/bin/pip install -r requirements-automl.txt

[python-3.12] -m venv bcrp_classify/.venv
bcrp_classify/.venv/bin/pip install -r bcrp_classify/requirements.txt
```

# Data
These workflows are configured to run using the 14 classification ADMET endpoints represented on the [vNN-ADMET web service](https://vnnadmet.bhsai.org). If you would like to run on alternative datasets, alter the contents of `data/dataset_config.csv`, and ensure your data exists at `data/raw/[dataset].csv` with columns `SMILES` and `Property`.

# Preprocessing

To preprocess the raw data, run the following:
``` Powershell
# Windows
.venv-preprocessing\Scripts\python.exe preprocessing\preprocess_all_endpoints.py
```
``` sh
# Linux
.venv-preprocessing/bin/python preprocessing/preprocess_all_endpoints.py
```
This will generate 2048-bit Morgan fingerprints and Mordred descriptors for the compounds in all datasets and separate them into a train-test scaffold split. The preprocessed data will appear under `data/preprocessed`.

# AutoML

To fit predictors using the frameworks run the following:

### Activate the virtual environment
``` Powershell
# Windows
.venv-automl\Scripts\Activate.ps1
```
``` sh
# Linux
source .venv-automl/bin/activate
```

### Variable-Nearest Neighbor Hyperparameter Optimization
``` sh
python vnn/vnn/vnn_all_endpoints_hpo.py
```

### AutoADMET Pipeline
``` sh
python autoadmet_pipeline/autoadmet_pipeline_all_endpoints_training.py
```

When running the AutoADMET pipeline for training, ensure that vNN is NOT included in the configuration's list of models. Find this configuration at `bcrp_classify\config.py`.

Both of the above scripts include options to skip the time-consuming hyperparameter search. For vNN, use `--use_prefit`. For the AutoADMET pipeline, use `--mode` with option `skip-cv` to skip the hyperparameter search, `load-model` to skip the ensembling and refitting the optimal model with full data, or `parse-results-only` to skip model evaluation and regenerate performance plots with previously collected performance data.

# Post processing and visualization

### Applicability domain performance
This script will calculate performance metrics inside and outside the applicability domain defined by the `DISTANCE_THRESHOLD` column in `dataset_config.csv`. Any inference compounds for which there is at least compound in the training set at least as Tanimoto-similar as `(1 - DISTANCE_THRESHOLD)` are considered inside the applicability domain; any inference compounds less similar than that threshold for all compounds in the training set are considered outside the applicability domain. For the 14 ADMET endpoints used, the `DISTANCE_THRESHOLD` values used are those posted in the vNN-ADMET web service under [About > Implemented ADMET Predictions](https://vnnadmet.bhsai.org/vnnadmet/availablemodels.xhtml).
The script outputs applicability domain performance metrics under `top_models/applicability_domain` and comparative plots under `visualization/out`.
``` sh
python top_models/applicability_domain/evaluate_applicability_domain.py
```

### Granular validation performance and significance tests
This script adds vNN validation performance to the AutoADMET validation performance record and reruns the AutoADMET validation performance plotting step. For each dataset, it will perform significance tests to compare the validation performance between vNN and the models that the AutoADMET pipeline trains. See the output at `output/autoadmet_pipeline/[DATASET]/cv_results/plots`.
``` sh
python visualization/plot_significance_tests.py
```

### Summarized validation performance
This script generates validation performance comparison plots between vNN and the best validating model from the AutoADMET pipeline across all datasets. See output at `visualization/out`.
``` sh
python visualization/plot_performance_validation.py
```

### Summarized test performance and time benchmarks
This script generates evaluation performance comparison plots between vNN, the best validating model from the AutoADMET pipeline, and its ensemble model. It also generates plots for time benchmark tests. See output at `visualization.out`.
``` sh
python visualization/plot_performance.py
```
