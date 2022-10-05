import numpy as np
import pandas as pd
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


def knn_impute(df: pd.DataFrame, n_neighbors: int = 5, nodata_values: Union[float, int] = np.nan,
               normalizer=MinMaxScaler()) -> pd.DataFrame:
    """
    Fill in missing values in a dataframe using K-Nearest Neighbors (KNN) imputation.
    Note: Using KNN imputation to fill in missing training set values can increase ML model performance.
    :param df: (pd.DataFrame) a dataframe with all numerical columns.
    :param n_neighbors: (int) number of neighbors to use in KNN imputation.
    :param nodata_values: (float or int) values to be interpreted as nodata (default=np.nan).
    :param normalizer: (sklearn.preprocessing class) a sklearn transformer flass capable of .fit() and .transform().
    :return: (pd.DataFrame) the KNN imputed version of the input dataframe.
    """
    try:
        normalizer = normalizer.fit(df.select_dtypes('numbers'))
        norm_np = normalizer.transform(df)
        knn = KNNImputer(missing_values=nodata_values, n_neighbors=n_neighbors)
        return pd.DataFrame(knn.fit_transform(norm_np), columns=df.columns, index=df.index)
    except Exception as e:
        print(f'Error: Exception arose - {e}')
