from typing import Callable, override

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import coo_array
from sklearn.utils.multiclass import unique_labels


def tanimoto_matrix(A: coo_array, B: coo_array) -> np.ndarray:
    """
    Given two sparse bit matrices, where each row (a bit vector) represents a
    single observation, each column a single feature, calculate the Tanimoto
    similarities between each observation of the first and each observation of the
    second.
    The matrices must have the same number of features/columns (n).
    The matrices may have different numbers of observations/rows.

    Args:
        A (coo_array): An axn sparse bit matrix with a observations and n features.
        B (coo_array): A bxn sparse bit matrix with b observations and n features.

    Returns:
        An axb matrix where cell i,j represents the Tanimoto similarity between row i of A and j of B.
    """
    # Here I refer to "cardinality" = "the number of on bits"

    # Make an axb matrix where cell i,j represents the cardinality of the intersection between row i of A and j of B.
    intersection = (A @ B.T).toarray()

    # Matrix where each cell i,_ is the cardinality of row i of A
    cardinalities_A = (np.zeros_like(intersection).T + A.sum(axis=1)).T

    # Matrix where each cell _,j is the cardinality of row j of B
    cardinalities_B = np.zeros_like(intersection) + B.sum(axis=1)

    # Make an axb matrix where cell i,j represents the cardinality of the union between row i of A and j of B.
    # Make this by adding the cardinalities of row i of A and j of B and subtracting their intersection.
    union = cardinalities_A + cardinalities_B - intersection

    # Return a matrix where cell i,j represents the cardinality of the intersection between row i of A and j of B as a
    # proportion of the cardinality of their union. As the union is a superset of the intersection, each cell must be [0,1].
    return intersection / union


def tanimoto_distance_matrix(A: coo_array, B: coo_array) -> np.ndarray:
    return 1 - tanimoto_matrix(A, B)


def v_neighbors_weighted_avg(
    dist_matrix: np.ndarray,
    y: np.ndarray,
    distance_threshold: float,
    smoothing_factor: float,
) -> np.ndarray:
    """
    Get the weighted averages of the v neighbors that pass the distance threshold for
    each query vector.

    Args:
        dist_matrix (np.ndarray): An (m x n) matrix of distances between m query
            vectors and n training vectors.
        y (np.ndarray): An (n x 1) vector of truth values for the n training vectors.
        distance_threshold (float): The distance threshold that training vectors must
            pass in order to be included in the weighted average for a query vector.
        smoothing_factor (float): The smoothing factor for the weight calculation.

    Returns:
        weighted_average (np.ndarray): An (m x 1) vector of weighted averages for each
            query vector. A weighted average will be NaN for a query vector if no
            training vectors passed the distance threshold.
    """
    # Mask the distances that fail the threshold with NaN
    applicable_dists = np.ma.masked_greater(
        x=dist_matrix,
        value=distance_threshold,
    ).filled(np.nan)

    # Turn the masked distances into weights
    weights: np.ndarray = np.exp(
        (-1 * np.power(applicable_dists / smoothing_factor, 2))
    )
    # Coalesce NaN to 0
    weights[np.isnan(weights)] = 0

    # Take the weighted average class value along training observations for each
    # query observation
    total_weighted_class_values = weights @ y
    total_weights = weights.sum(axis=1)
    # Avoid divide by zero warnings. If a row has no nonzero weights, that
    # observation is outside the applicability domain and its prediction will
    # be NaN.
    total_weights[total_weights == 0] = np.nan
    weighted_average = total_weighted_class_values / total_weights

    return weighted_average


class VariableNearestNeighborsRegressor:
    __distance_threshold: float
    __smoothing_factor: float

    __X: coo_array
    __y: np.ndarray
    __calc_distance_matrix: Callable[[coo_array, coo_array], np.ndarray]

    def __init__(
        self,
        smoothing_factor: float,
        distance_threshold: float = 1,
        calc_distance_matrix: Callable[
            [coo_array, coo_array],
            np.ndarray,
        ] = tanimoto_distance_matrix,
    ) -> None:
        """
        Args:
            smoothing_factor (float): A smoothing factor for making weights. Higher
                smoothing factors dampen the distance penalty.
            distance_threshold (float): The maximum distance between observations.
                Training observations with larger distances for a query observation
                will not be included in the weighted average class prediction for
                that query. As Tanimoto distance ranges from 0 to 1, a value of 1
                will consider all training data for every query observation and all
                query observations will receive predictions.
            calc_distance_matrix (Callable[[coo_array, coo_array], np.ndarray]): A
                function that creates an (m x n) matrix of distances between m query
                vectors and n training vectors given an (m x k) query vector matrix and
                (n x k) training vector matrix.
        """
        self.__smoothing_factor = smoothing_factor
        self.__distance_threshold = distance_threshold
        self.__calc_distance_matrix = calc_distance_matrix

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        self.__X = coo_array(X)
        self.__y = np.array(y)

    def predict(self, X: ArrayLike) -> np.ndarray:
        # Get the tanimoto distances of each query observation to each training observation
        dist_matrix = self.__calc_distance_matrix(coo_array(X), self.__X)

        return v_neighbors_weighted_avg(
            dist_matrix,
            self.__y,
            self.__distance_threshold,
            self.__smoothing_factor,
        )


class VariableNearestNeighborsClassifier(VariableNearestNeighborsRegressor):
    @override
    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        super().fit(X, y)
        self.classes_ = unique_labels(y)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return super().predict(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        class_predictions = self.predict_proba(X).round()
        return class_predictions
