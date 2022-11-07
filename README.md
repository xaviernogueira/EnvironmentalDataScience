🌎 Environmenta Data Science 🌍
==============================
🚧🚧🚧🚧 WORK IN PROGRESS 🚧🚧🚧🚧

**Roadmap:**
1. ~~Create generalizable ML Pipeline Protocols in `protocols.py`.~~
2. ~~Create `CatBoostML.train_regression()` to automated regression model training and evaluation.~~
3. Create a `CatBoostML.train_classifier()` class that automates classification model training and evaluation.


👨🏽‍💻 **Author:** [Xavier Rojas Nogueira](https://www.linkedin.com/in/xavier-r-nogueira-286819120/)

🏜 **Descrpition:** A place for my tools, workflows, side projects, etc. related to environmental data science. 

# Repository Contents

## `\machine_learning\` 🤖 
*  ### `preprocessing_functions.py`
    * ```python 
      csvs_to_parquets(data_dir: str, ignore_cols: list = None, to_datetime_cols: dict = None, out_path: str = None) -> str
            """Converts very large CSV files into partitioned performance oriented .parquet files in parallel using dask."""
        ```    
        * `:param data_dir:` (str path) a valid directory path containing >=1.csv files.
        * `:param ignore_cols:` (list of str column headers, optional dtype=object columns to NOT convert to category.
        * `:param to_datetime_cols:` (list of str column headers, optional) columns to convert to datetime.
        * `:param out_path:` (str path, optional) a path to save the output partitioned .parquet files.
        * `:return:` (str path) out_path where the parquet files are stored.
    
    * ```python
      knn_impute(df: pd.DataFrame, n_neighbors: int = 5, nodata_values: Union[float, int] = np.nan,
               normalizer=MinMaxScaler()) -> pd.DataFrame:
        """Fill in missing values in a dataframe using K-Nearest Neighbors (KNN) imputation. Note: Using KNN imputation to fill in missing training set values can increase ML model performance."""
      ```

        * `:param df:` (pd.DataFrame) a dataframe with all numerical columns.
        * `:param n_neighbors:` (int) number of neighbors to use in KNN * imputation.
        * `:param nodata_values:` (float or int) values to be interpreted as nodata (default=np.nan).
        * `:param normalizer:` (sklearn.preprocessing class) a sklearn transformer flass capable of .fit() and .transform().
        * `:return:` (pd.DataFrame) the KNN imputed version of the input dataframe.

* ### `catboost_ml.py`

## 🧙 `\sandbox` - a directory to store expiremental work, markdown notes files, and half-polished Jupyter Notebooks.

