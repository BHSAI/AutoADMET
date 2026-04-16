from typing import Final, Literal

import numpy as np
from scipy.sparse import coo_array
from scipy import sparse
from scipy import stats
import vnn.variable_nearest_neighbor as vnn_classifier
import pandas as pd
import itertools as it
import sklearn.metrics
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from chembl_structure_pipeline import standardizer
from pathlib import Path

N_FOLDS = 10

CLASSIFICATION: Final = "classification"
REGRESSION: Final = "regression"
PROBLEM_TYPE = Literal["classification", "regression"]

NAME_COL: Final = "name"
SMILES_COL: Final = "smiles"
PROPERTY_COL: Final = "property"


def eval_classification_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    total_original = len(y_true)

    out_of_domain = np.isnan(y_pred)
    y_true = y_true[~out_of_domain]
    y_pred = y_pred[~out_of_domain]
    num_out_of_domain = out_of_domain.sum()
    coverage = 1 - num_out_of_domain / total_original

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel().tolist()

    total_pos = tp + fn
    total_neg = tn + fp
    total_within_domain = total_pos + total_neg

    sensitivity = tp / total_pos if total_pos else np.nan
    specificity = tn / total_neg if total_neg else np.nan
    accuracy = (tp + tn) / total_within_domain if total_within_domain else np.nan
    if total_within_domain:
        e = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (
            np.pow(total_within_domain, 2)
        )
        kappa = (accuracy - e) / (1 - e)
    else:
        kappa = np.nan
    tpr = tp / total_pos if total_pos else np.nan
    fpr = fp / total_neg if total_neg else np.nan
    roc = 0.5 * (1 - fpr + tpr) if total_pos and total_neg else np.nan

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "kappa": kappa,
        "Coverage": coverage,
        "AreaUnderTheCurve": roc,
    }


def eval_regression_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    total_original = len(y_true)

    out_of_domain = np.isnan(y_pred)
    y_true = y_true[~out_of_domain]
    y_pred = y_pred[~out_of_domain]
    num_out_of_domain = out_of_domain.sum()
    coverage = 1 - num_out_of_domain / total_original

    correlation = stats.pearsonr(y_true, y_pred).correlation

    return {
        "R": correlation,
        "Coverage": coverage,
    }


eval_metrics = {
    CLASSIFICATION: eval_classification_dict,
    REGRESSION: eval_regression_dict,
}


def v_neighbors_weighted_avg_oof(
    dist_matrix: np.ndarray,
    y: np.ndarray,
    distance_threshold: float,
    smoothing_factor: float,
    n_folds: int,
) -> np.ndarray:
    """
    Get the out-of-fold weighted averages of the v neighbors that pass the distance
    threshold for each query vector. Before predicting probabilities, each distance
    (i, j) where query vector i and training vector j are in the same fold
    (i.e., i %% n_folds == j %% n_folds) will be masked to NaN such that training
    vector j will be excluded from the weighted_average for query vector i.

    Args:
        dist_matrix (np.ndarray): An m x n matrix representing the
            distances between m query vectors and n training vectors.
        y (np.ndarray): An n x 1 matrix representing the truth values for the n
            training vectors.
        distance_threshold (float): The distance threshold.
        smoothing_factor (float): The smoothing factor.
        n_folds (int): The number of cross validation folds. Folds are assigned by
            the index of the vector mod n_folds.
    """
    # Get an n_folds x n_folds identity matrix and tile it to the size of the distance matrix.
    # Every cell (i, j) where i % n_folds == j % n_folds will be True.
    shape = np.array(dist_matrix.shape)
    same_fold = np.tile(
        np.eye(n_folds).astype(bool),
        tuple(np.ceil(shape / n_folds).astype(int)),
    )[: shape[0], : shape[1]]

    oof_tanimoto_dist = np.ma.masked_array(dist_matrix, same_fold).filled(np.nan)

    return vnn_classifier.v_neighbors_weighted_avg(
        dist_matrix=oof_tanimoto_dist,
        y=y,
        distance_threshold=distance_threshold,
        smoothing_factor=smoothing_factor,
    )


def run_grid_search(
    X: coo_array,
    y: np.ndarray,
    problem_type: PROBLEM_TYPE = CLASSIFICATION,
    smoothing_factor_space_resolution: int = 10,
    tanimoto_threshold_space_resolution: int = 10,
    n_folds=N_FOLDS,
) -> pd.DataFrame:
    """
    Perform (smoothing_factor_space_resolution x tanimoto_threshold_space_resolution) grid search on
    smoothing factor and tanimoto distance threshold and report the evaluation metrics.

    Args:
        X (pd.DataFrame): An (n x k) dataframe where each row is a training vector and each column a feature.
        y (np.ndarray): An (n x 1) vector of truth values for the training vectors.
        problem_type (Literal["classification", "regression"]): The problem type.
        smoothing_factor_space_resolution (int): The resolution of the search space for smoothing factor. Will search this many values within the range (0, 1].
        tanimoto_threshold_space_resolution (int): The resolution of the search space for distance threshold. Will search this many values within the range (0, 1].

    Returns:
        A dataframe of the evaluated combinations of hyperparameters and their validation performance metrics.
    """
    tanimoto_dist = vnn_classifier.tanimoto_distance_matrix(X, X)

    # Perform grid search over smoothing factor and distance threshold
    smoothing_factor_space = (
        (x + 1) / smoothing_factor_space_resolution
        for x in range(smoothing_factor_space_resolution)
    )
    tanimoto_threshold_space = (
        (x + 1) / tanimoto_threshold_space_resolution
        for x in range(tanimoto_threshold_space_resolution)
    )
    search_grid = it.product(smoothing_factor_space, tanimoto_threshold_space)
    results = []
    for smoothing_factor, tanimoto_threshold in tqdm(
        search_grid,
        "Testing hyperparameters",
        smoothing_factor_space_resolution * tanimoto_threshold_space_resolution,
    ):
        y_pred = v_neighbors_weighted_avg_oof(
            dist_matrix=tanimoto_dist,
            y=y,
            distance_threshold=tanimoto_threshold,
            smoothing_factor=smoothing_factor,
            n_folds=n_folds,
        )
        if problem_type == CLASSIFICATION:
            y_pred = y_pred.round()

        results.append(
            {
                "SmoothFactor": smoothing_factor,
                "TanimotoDistance": tanimoto_threshold,
                **eval_metrics[problem_type](y, y_pred),
            }
        )

    return pd.DataFrame(results)


def standardize_smiles(smiles: str) -> str | None:
    """
    Standardize a molecule. Choose the largest fragment, de-salt,
    Chembl-standardize, and strip stereochemistry.

    Parameters
    ----------
    smiles (str) : A SMILES string representation of the molecule to standardize.

    Returns
    -------
    A SMILES string representation for the standardized molecule, if the given
    SMILES string represents a valid molecule. None if the given SMILES string
    is invalid.
    """

    if pd.isna(smiles) or len(smiles) == 0:
        return None

    # Read in molecule, sanitize
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    # Choose largest fragment
    largest_Fragment = rdMolStandardize.LargestFragmentChooser()
    molecule = largest_Fragment.choose(molecule)

    # Remove salts
    molecule = standardizer.get_parent_mol(molecule)[0]

    # Use Chembl standardizer
    molecule = standardizer.standardize_mol(molecule)

    return Chem.MolToSmiles(molecule)


def create_fingerprint_matrix(smiles: pd.Series) -> tuple[coo_array, pd.Series]:
    """
    Given a list of raw SMILES, standardize them and turn them into molecular fingerprints.
    Note that any invalid SMILES will be mapped to all-zero fingerprints so that the resulting
    matrix is well-formed.

    Args:
        smiles (pd.Series): A series of SMILES strings.

    Returns
    -------
        fingerprint_matrix (coo_array): A sparse matrix where each row is the molecular
            fingerprint for the corresponding entry in the input SMILES series.
        standardized_smiles (pd.Series): A series where each entry is the standardized SMILES
            for the corresponding entry in the input SMILES series.
    """
    # Initialize fingerprint generator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    # Standardize SMILES
    standardized_smiles = smiles.apply(standardize_smiles)
    # Generate fingerprints
    molecule = standardized_smiles.fillna("").apply(Chem.MolFromSmiles)
    fingerprint = molecule.apply(mfpgen.GetFingerprintAsNumPy)
    fingerprint_matrix = coo_array(np.stack(fingerprint))  # type: ignore
    return fingerprint_matrix, standardized_smiles


def generate_reference_data(
    reference_df: pd.DataFrame,
    invert_class: bool = False,
) -> tuple[coo_array, np.ndarray]:
    """
    Given a reference data dataframe with SMILES and Class columns, generate the fingerprint
    matrix and values vector.

    Args:
        reference_df (pd.DataFrame): The dataframe for the reference dataset.
        invert_class (bool): If true, flip the class labels. Only makes sense for binary
            classification datasets.

    Returns
    -------
        X_reference (coo_array): A sparse bit matrix where each row is the fingerprint for
            the corresponding SMILES in the reference dataset.
        y_reference (np.ndarray): A vector of values for each molecule in the reference
            dataset. For binary classification {0, 1}, for regression a real number.
    """
    og_smiles = reference_df[SMILES_COL]
    values = reference_df[PROPERTY_COL]
    if invert_class:
        values = 1 - values

    X_reference, _ = create_fingerprint_matrix(og_smiles)
    y_reference = values.to_numpy()

    return X_reference, y_reference


def get_reference_data(
    endpoint: str,
    data_folder: str,
    is_classification: bool = True,
    invert_class: bool = False,
) -> tuple[coo_array, np.ndarray]:
    """
    Load the reference data for the provided endpoint or generate it from the raw dataset if it has not yet been generated.

    Args:
        endpoint (str): The name of the reference dataset.
        invert_class (bool): If true, flip the class labels. Only makes sense for binary
            classification datasets.

    Returns
    -------
        X_reference (coo_array): A sparse bit matrix where each row is the fingerprint for
            the corresponding SMILES in the reference dataset.
        y_reference (np.ndarray): A vector of values for each molecule in the reference
            dataset. For binary classification {0, 1}, for regression a real number.
    """
    numpy_cache_dir = f"{data_folder}/numpy_cache"
    X_reference_file = f"{numpy_cache_dir}/{endpoint}.fp.npz"
    y_reference_file = f"{numpy_cache_dir}/{endpoint}.y.npy"

    import os

    if os.path.isfile(X_reference_file) and os.path.isfile(y_reference_file):
        X_reference = sparse.load_npz(X_reference_file).astype(np.int8)
        y_reference = np.load(y_reference_file).astype(
            np.int8 if is_classification else float
        )
    else:
        reference_df = pd.read_csv(f"{data_folder}/{endpoint}.csv")
        reference_df.columns = reference_df.columns.map(str.lower)
        X_reference, y_reference = generate_reference_data(reference_df, invert_class)
        Path(numpy_cache_dir).mkdir(exist_ok=True)
        sparse.save_npz(X_reference_file, X_reference)
        np.save(y_reference_file, y_reference)

    return X_reference, y_reference


def run_admet_predictions(
    query_df: pd.DataFrame,
    config_df: pd.DataFrame,
    data_folder: str,
) -> pd.DataFrame:
    """
    Run vNN predictions for all ADMET datasets.

    Args:
        query_df (pd.DataFrame): A dataframe of the query data with "Name" and "SMILES"
            columns.

    Returns:
        pred_df (pd.DataFrame): A copy of the provided dataframe with a standardized
            SMILES column and high- and low-confidence prediction columns for each
            of the ADMET endpoints.

    """

    # Prepare the return dataframe
    pred_df = query_df[[NAME_COL, SMILES_COL]].copy()
    pred_df.columns = ["name", "Original_Smiles"]
    X_query, pred_df["SMILES"] = create_fingerprint_matrix(pred_df["Original_Smiles"])
    invalid_smiles_idx = pred_df["SMILES"].isna()

    # Make predictions for the high confidence and low confidence constraints for each endpoint
    for endpoint, config in config_df.iterrows():
        # Load cached fingerprint matrix and values for endpoint
        X_reference, y_reference = get_reference_data(
            endpoint=endpoint,  # type: ignore
            is_classification=config["isClz"],
            invert_class=config["inverse"],
            data_folder=data_folder,
        )

        # Make tanimoto distance matrix
        tanimoto_dist = vnn_classifier.tanimoto_distance_matrix(X_query, X_reference)

        # Make high confidence predictions
        y_pred_high_conf = vnn_classifier.v_neighbors_weighted_avg(
            dist_matrix=tanimoto_dist,
            y=y_reference,
            distance_threshold=config["tdhc"],
            smoothing_factor=config["sfhc"],
        )

        # Make low confidence predictions
        y_pred_low_conf = vnn_classifier.v_neighbors_weighted_avg(
            dist_matrix=tanimoto_dist,
            y=y_reference,
            distance_threshold=config["tdlc"],
            smoothing_factor=config["sflc"],
        )

        # Invalid molecules are represented by all-zero fingerprints in X_query, so
        # they will not have high confidence predictions, but will have low
        # confidence predictions. We don't want them to have either.
        y_pred_low_conf[invalid_smiles_idx] = np.nan

        if config["isClz"]:
            y_pred_high_conf = y_pred_high_conf.round()
            y_pred_low_conf = y_pred_low_conf.round()
        else:
            # Since the only regression problem is MRTD, do a transformation specific to MRTD
            mol_weight = pred_df["SMILES"].fillna("").apply(Chem.MolFromSmiles).apply(Descriptors.ExactMolWt)  # type: ignore
            y_pred_high_conf = mol_weight * np.power(10, y_pred_high_conf) * 1000 * 60
            y_pred_low_conf = mol_weight * np.power(10, y_pred_low_conf) * 1000 * 60
            # Fill in missing high confidence predictions with low confidence predictions
            nan_idx = np.isnan(y_pred_high_conf)
            y_pred_high_conf[nan_idx] = y_pred_low_conf[nan_idx]

        pred_df[endpoint] = y_pred_high_conf
        pred_df[f"{endpoint}_LC"] = y_pred_low_conf

    return pred_df


def build_model(
    reference_df: pd.DataFrame,
    problem_type: PROBLEM_TYPE = CLASSIFICATION,
) -> pd.DataFrame:
    """
    Given a compound dataset as dataframe of raw SMILES and reference values, perform vNN
    grid search to find the highest cross-validation performing distance threshold and
    smoothing factor hyperparameters.

    Args:
        reference_df (DataFrame): A compound reference dataset dataframe with columns "SMILES" and "Class".
        problem_type (Literal["classification", "regression"]): The problem type.

    Returns:
        A dataframe of the evaluated combinations of hyperparameters and their validation performance metrics.
    """
    X_reference, _ = create_fingerprint_matrix(reference_df[SMILES_COL])
    y_reference = reference_df[PROPERTY_COL].to_numpy()

    return run_grid_search(X_reference, y_reference, problem_type)


def run_model(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    distance_threshold: float,
    smoothing_factor: float,
    problem_type: PROBLEM_TYPE = CLASSIFICATION,
) -> pd.DataFrame:
    """
    Args:
        query_df (DataFrame): A query compound dataframe with columns "Name" and "SMILES"
        reference_df (DataFrame): A compound reference dataset dataframe with columns "SMILES" and "Class".
        distance_threshold (float): The distance threshold for the vNN algorithm.
        smoothing_factor (float): The smoothing factor for the vNN algorithm.
        problem_type (Literal["classification", "regression"]): The problem type.

    Returns:
        pred_df (DataFrame): A dataframe containing, for each query compound, the given
            name, the given SMILES, the standardized SMILES, and the property prediction.
    """

    # Prepare the return dataframe
    pred_df = query_df[[NAME_COL, SMILES_COL]].copy()
    pred_df.columns = ["name", "Original_Smiles"]
    X_query, pred_df["SMILES"] = create_fingerprint_matrix(pred_df["Original_Smiles"])
    invalid_smiles_idx = pred_df["SMILES"].isna()

    X_reference, _ = create_fingerprint_matrix(reference_df[SMILES_COL])
    y_reference = reference_df[PROPERTY_COL].to_numpy()

    tanimoto_dist = vnn_classifier.tanimoto_distance_matrix(X_query, X_reference)

    y_pred = vnn_classifier.v_neighbors_weighted_avg(
        dist_matrix=tanimoto_dist,
        y=y_reference,
        distance_threshold=distance_threshold,
        smoothing_factor=smoothing_factor,
    )
    y_pred[invalid_smiles_idx] = np.nan
    if problem_type == CLASSIFICATION:
        y_pred = y_pred.round()

    pred_df["Prediction"] = y_pred

    return pred_df
