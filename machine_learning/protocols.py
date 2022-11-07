import pandas as pd
import numpy as np
from typing import Protocol, List, Union, Tuple
from abc import abstractmethod

class DataImputer(Protocol):
    """Abstract base class signature of a missing value imputation routine"""
    @abstractmethod
    def fill_nans(self, ignore_cols: List[str] = None) -> pd.DataFrame:
        # note we can essentially wrap ML into this or a more basic algo like KNN
       pass

class PreProcessRoutine(Protocol):
    """Abstract base class signature of a data pre-processing pipeline"""
    def prep_data(in_data: pd.DataFrame) -> pd.DataFrame:
        pass

class MachineLearningModel(Protocol):
    """Abstract base class signature of a ML model supporting regression + classification"""
    
    @abstractmethod
    def train_regressor(self,
                    features: Union[np.ndarray, pd.DataFrame],
                    target: Union[np.ndarray, pd.Series],
                    hyperparams: dict = None) -> Tuple(object, dict):
       pass

    @abstractmethod
    def train_classifier(self,
                        features: pd.DataFrame, target: pd.Series) -> object:
       pass

class ParameterSearcher(Protocol):
    """Abstract base class signature of a parameter searching algo (i.e. Grid Search)"""
    @abstractmethod
    def search(self, bounds: dict,
                grid_steps: dict = None, manual_grid: dict = None) -> dict:
       pass
