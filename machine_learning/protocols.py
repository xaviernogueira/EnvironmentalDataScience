import pandas as pd
import numpy as np
from typing import Protocol, List, Union, Tuple, Dict
import abc


class DataImputer(abc.ABC):
    """Abstract base class signature of a missing value imputation routine"""
    @abc.abstractmethod
    def fill_nans(self, ignore_cols: List[str] = None) -> pd.DataFrame:
        # note we can essentially wrap ML into this or a more basic algo like KNN
        pass


class PreProcessRoutine(abc.ABC):
    """Abstract base class signature of a data pre-processing pipeline"""
    def prep_data(in_data: pd.DataFrame) -> pd.DataFrame:
        pass


class MachineLearningModel(abc.ABC):
    """Abstract base class signature of a ML model supporting regression + classification"""

    @abc.abstractmethod
    def regression_kfold_evaluation(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        regressor_params: dict,
        k_folds: int = 10,
    ) -> Dict[str, List[float]]:
        pass

    @abc.abstractmethod
    def train_regressor(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.Series],
        hyperparams: dict = None
    ) -> object:
        pass

    @abc.abstractmethod
    def train_classifier(self,
                         features: pd.DataFrame, target: pd.Series) -> object:
        pass


class ParameterSearcher(abc.ABC):
    """Abstract base class signature of a parameter searching algo (i.e. Grid Search)"""
    @abc.abstractmethod
    def search(self, bounds: dict,
               grid_steps: dict = None, manual_grid: dict = None) -> dict:
        pass
