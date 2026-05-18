from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

# ----------------------------
# Data containers
# ----------------------------

@dataclass
class RTMData:
    X: np.ndarray
    Y_flat: np.ndarray
    wavelengths: np.ndarray
    param_names: list[str]
    function_names: list[str]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


@dataclass
class SplitRTMData:
    X_tr: np.ndarray
    X_val: np.ndarray
    Y_tr: np.ndarray
    Y_val: np.ndarray


@dataclass
class FunctionModel:
    reducer: Any
    y_scaler: Any
    gpr: GaussianProcessRegressor


@dataclass
class PipelineConfig:
    n_components: int = 10
    dimred_method: Literal["pca", "kpca"] = "pca"
    scale_type: Literal["standard", "minmax"] = "minmax"
    n_restarts_optimizer: int = 5
    random_state: int = 42