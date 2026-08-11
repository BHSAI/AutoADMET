from typing import override

from numpy import ndarray
import pandas as pd

from autogluon.core.models import AbstractModel


class VNNModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ):

        import vnn.variable_nearest_neighbor as vnn_lib

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        self.model = vnn_lib.VariableNearestNeighborsClassifier(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {
            "smoothing_factor": 0.3,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=["int"],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    
    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {
            "valid_stacker": False,
        }
        default_ag_args.update(extra_ag_args)
        return default_ag_args
