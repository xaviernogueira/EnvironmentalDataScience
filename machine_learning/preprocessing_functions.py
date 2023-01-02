import gc
from pathlib import Path
import numpy as np
import pandas as pd
import dask.dataframe as dd
from typing import Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from typing import Dict, List, Optional


def knn_impute(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    nodata_values: Union[float, int] = np.nan,
    normalizer=MinMaxScaler(),
) -> pd.DataFrame:
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


def _id_csvs(data_dir: Union[Path, str]) -> dict:
    """
    Takes a dir path containing .csv tables, and returns a dict w/ the .csv names as keys, and paths as values.
    :param data_dir: (str path) a valid directory path containing >=1 .csv files.
    :return: dict of the form {'big_data': 'C:\\path_to_data\\big_data.csv', ...}.
    """

    # handle path input
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    elif not isinstance(data_dir, Path):
        raise TypeError(
            'param:data_dir must be a pathlib.Path object or a string path!')
    if not data_dir.is_dir():
        raise ValueError(
            f'param:data_dir = {data_dir} is not a directory or does not exist! ')

    csv_dict = {}

    # iterate over data_dir looking for .csv files
    for file in data_dir.iterdir():
        if file.suffix == '.csv':
            csv_dict[file.name.replace('.csv', '')] = file

    # print csv_dict items, and return the dictionary
    print(f'csv_dict initiated {csv_dict.keys()}')
    return csv_dict


def _import_csv(
    csv_dict: Dict[str, Path],
    index_col: Optional[str] = None,
) -> Dict[str, dd.DataFrame]:
    """
    Uses dask to partition into multiple dask.dataframes with max # of rows = chunk_size
    :param csv_dict: the output of id_csv
    :param index_col: (optional) index column for the csv
    :returns: csv_dict, but updated so that each key stores a dask dataframe
    """
    for data_name in csv_dict.keys():
        print(f'Importing {data_name}...')
        csv_path = csv_dict[data_name]

        # read the csv sampling 5MB at a time to determine dtypes
        df = dd.read_csv(csv_path, engine='c', sample=5000000)

        # if an index is provided, warn the user about the performance penalty
        if index_col is not None:
            print(
                f'WARNING: Setting index to {index_col} will drastically hurt performance!'
            )
            df = df.set_index(index_col)

        # update the dictionary and collect trash
        sub_dict = {data_name: df}
        csv_dict.update(sub_dict)
        gc.collect()
    return csv_dict


def _object_to_cat(
    df: dd.DataFrame,
    ignore_cols: Optional[List[str]] = None,
    override_cols: Optional[List[str]] = None,
) -> dd.DataFrame:
    """
    Converts all object columns to category type in dask.dataframe, unless specified in ignore_cols
    :param df: (dask.DataFrame) as dask dataframe.
    :param ignore_cols: (list of str column headers, optional) list of object columns to NOT get converted to category.
    :param override_cols: (list of str column headers, optional) manually converts a subset of columns to category.
    :return: (dask.DataFrame) the dask dataframe with columns converted.
    """
    if ignore_cols is None:
        ignore_cols = []
    if not override_cols:
        object_cols = df.select_dtypes(include='object')
    else:
        object_cols = override_cols

    # for each object column, convert the original dataframes dtype
    for col in object_cols:
        try:
            if str(col) not in ignore_cols:
                df[col] = df[col].astype('category')
        except KeyError or TypeError:
            print(f'ERROR: Could not convert column {col} to category')
    return df


def csvs_to_parquets(
    data_dir: Union[str, Path],
    ignore_cols: Optional[List[str]] = None,
    to_datetime_cols: Optional[Dict[str, str]] = None,
    out_dir: Union[str, Path] = None,
) -> str:
    """
    Converts very large CSV files into partitioned .parquet files in parallel using dask.
    :param data_dir: (str path) a valid directory path containing >=1 .csv files.
    :param ignore_cols: (list of str column headers, optional) dtype=object columns to NOT convert to category.
    :param to_datetime_cols: (list of str column headers, optional) columns to convert to datetime.
    :param out_dir: (str path, optional) a path to save the output partitioned .parquet files.
    :return: (str path) out_dir where the parquet files are stored.
    """
    # get all csvs from param:data_dir a dictionary
    csv_dict = _id_csvs(data_dir)

    # update csv_dict to store dask dataframes for each data type
    _import_csv(csv_dict=csv_dict)
    _ = gc.collect()
    print(
        f'csv_dict stores dask.dataframes with the following keys: {csv_dict.keys()}')

    # convert all other object columns to category, and any columns to datetime using a format string
    for name, df in csv_dict.items():
        if to_datetime_cols is not None:
            for col, format_str in to_datetime_cols.items():
                try:
                    df[col] = dd.to_datetime(df[col], format=format_str)
                except Exception as e:
                    print(e)
        csv_dict[name] = _object_to_cat(df, ignore_cols=ignore_cols)
    _ = gc.collect()

    # define the output path automatically is param:out_dir is not valid/provided
    if out_dir is not None:
        if isinstance(out_dir, str):
            out_dir = Path(out_dir)
        elif not isinstance(out_dir, Path):
            raise TypeError(
                'param:out_dir must be a pathlib.Path object or a string!'
            )

    if not out_dir or not out_dir.exists():
        out_dir = Path(data_dir / 'output_parquets')

    for title, df in csv_dict.items():
        out_parq = out_dir / f'{title}.parquet'

        # convert the dask dataframe to parquet
        dd.to_parquet(
            df,
            out_parq,
            engine='pyarrow',
            compression='gzip',
        )
        print(f'{title}.parquet saved @ {out_dir}')

        # collect garbage and delete the previous dask dataframe
        _ = gc.collect()
        del df

    print(f'Done! all .parquet files saved @ {out_dir}')
    return out_dir
