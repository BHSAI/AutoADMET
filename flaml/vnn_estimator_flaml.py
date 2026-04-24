from flaml.automl.model import SKLearnEstimator
from flaml import tune


class VNNEstimator(SKLearnEstimator):
    def __init__(self, task="binary", n_jobs=None, **config):
        super().__init__(task, **config)

        if isinstance(task, str):
            from flaml.automl.task.factory import task_factory

            task = task_factory(task)

        if task.is_classification():
            from vnn.variable_nearest_neighbor_classifier import (
                VariableNearestNeighborsClassifier,
            )

            self.estimator_class = VariableNearestNeighborsClassifier
        else:
            from vnn.variable_nearest_neighbor_classifier import (
                VariableNearestNeighborsRegressor,
            )

            self.estimator_class = VariableNearestNeighborsRegressor

    @classmethod
    def search_space(cls, data_size, task):
        space = {
            "smoothing_factor": {
                "domain": tune.uniform(lower=0.1, upper=1.0),
                "init_value": 0.3,
            },
        }
        return space
