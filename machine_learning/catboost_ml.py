import catboost as cb
from catboost import CatBoostRegressor, CatBoostClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn as skl
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

# import type hints
from typing import Union, Tuple, List

# import protocol to implement
from protocols import MachineLearningModel

# functions to move to preprocessing_functions.py


def categorize_features(
    features: pd.DataFrame,
    unique_limit: int = 20,
) -> pd.DataFrame:
    """
    Converts all catgeory features to dtype=category.
    :param features: (pd.DataFrame) features as columns in a dataframe.
    :param unique_list: (int) if the number of unique values in a column is < param:unique_limit, it is considered categorical.
    :returns: (list) a list of categorical column headers or index positions.
    """
    # id all non-numeric columns as categorical
    obj_cols = features.select_dtypes(include='object').columns
    features[obj_cols] = features[obj_cols].astype('category')

    # convert integer columns to category depending on the number of unique values
    int_cols = features.select_dtypes(include='int')
    for col in int_cols:
        if len(list(features[col].unique())) < unique_limit:
            features[col] = features[col].astype('category')

    return features


def prep_training_inputs(
    features: Union[np.ndarray, pd.DataFrame],
    target: Union[np.ndarray, pd.Series],
    allow_index_join: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:

    # if numpy inputs, convert to pandas
    if isinstance(features, np.ndarray):
        headers = [f'f{i}' for i in range(features.shape[1])]
        features = pd.DataFrame(
            features.squeeze(),
            columns=headers,
        )

    # convert columns to category dtype as appropriate
    features = categorize_features(features)

    if isinstance(target, np.ndarray):
        target = pd.Series(target.squeeze())
        target.name = 'target'
        # if multiple target columns are passed in, return an error
        if len(target.shape) > 1:
            return print('ERROR: Target must be a nx1 array, two columns were passed in.')

    # verify that the shapes match
    if not features.shape[0] == target.shape[0]:
        print(f'Warning: Features and target data have different # of rows')
        if features.index.name == target.index.name and allow_index_join:
            print(f'Joining on matching index: {features.index.name}. '
                  'Rows without both target and training data will be removed!')
            joined = features.join(target, how='inner')
            features = joined.drop(columns=target.name)
            target = pd.Series(joined[target.name])
            del joined
        else:
            return print('ERROR: Features and Target are different lengths + no matching index found')
    return (features, target)


class CatBoostML(MachineLearningModel):
    # CatBoost specific extension functions
    def __init__(self,
                 performance_log: dict = {},
                 ) -> None:
        self.performance_log = performance_log
        self.parameter_log = {}
        self.true_vs_predicted = {}
        self.run = 0

    def clear_log(self) -> None:
        self.performance_log = {}
        self.true_vs_predicted = {}
        self.run = 0

    def clean_model_params(
        self,
        model_type: str,
        params_dict: dict,
    ) -> dict:
        """
        Returns the hyperparams dictionary removing any hyperparams kwargs not
        applicable to CatBoost.
        :param model_type: (str) either regressor or classifier.
        """
        out_dict = {}

        for key, value in params_dict.items():
            try:
                if model_type == 'regressor':
                    cb.to_regressor(cb.CatBoost(params={key: value}))
                elif model_type == 'classifier':
                    cb.to_classifier(cb.CatBoost(params={key: value}))
                out_dict.update({key: value})
            except cb.CatBoostError as e:
                print(
                    f'CatBoostError: Could not set {key}:{value} CatBoostRegressor param')
                print(str(e))
        return out_dict

    # MachineLearningModel protocol function implementations
    def get_model_outputs(
        self,
        train_r2s: list,
        test_r2s: list,
        train_maes: list,
        test_maes: list,
    ) -> dict:
        """
        A function that returns a list containing mean and STD R2 and MAE values from 10-K fold CV
            for a given dependent variable's Regression model.
        :returns: a list containing 
            [depend_col (str),
            mean_train_r2,
            mean_test_r2,
            std_train_r2,
            std_test_r2,
            mean_train_mae,
            mean_test_mae,
            std_train_mae,
            std_test_mae]
        """
        # convert train data to arrays
        np_train_r2s = np.array(train_r2s)
        np_train_maes = np.array(train_maes)

        # convert test data to arrays
        np_test_r2s = np.array(test_r2s)
        np_test_maes = np.array(test_maes)

        # create out_list with model performance metrics
        out_dict = {
            'mean_train_r2': np_train_r2s.mean(),
            'std_train_r2': np_train_r2s.std(),
            'mean_train_mae': np_train_maes.mean(),
            'std_train_mae': np_train_maes.std(),
            'mean_test_r2': np_test_r2s.mean(),
            'std_test_r2': np_test_r2s.std(),
            'mean_test_mae': np_test_maes.mean(),
            'std_test_mae': np_test_maes.std(),
        }

        out_dict['train_test_gap'] = out_dict['mean_train_r2'] - \
            out_dict['mean_test_r2']
        return out_dict

    def regression_kfold_evaluation(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        regressor_params: dict,
        cat_features: list = None,
        k_folds: int = 10,
    ) -> dict:
        """Creates the main performance evaluation dictionary"""
        # init kfolds
        kfolds = KFold(n_splits=k_folds)

        # get cat_features list
        if cat_features is None:
            cat_features = list(features.select_dtypes('category').columns)

        # make lists to store performance metrics
        train_r2s = []
        train_maes = []
        test_r2s = []
        test_maes = []

        # make two list to store true and predicted values (in order to make performance charts)
        true_arrays = []
        predicted_arrays = []

        for train_index, test_index in kfolds.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = target.iloc[train_index].to_numpy(
            ), target.iloc[test_index].to_numpy()

            # build regression or classification model (uses Logloss optimization for binary classification data and Accuracy eval_metric)
            cb_model = cb.to_regressor(cb.CatBoost(params=regressor_params))
            cb_model = cb_model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
                verbose=False,
            )

            # make a prediction using the training data
            train_preds = cb_model.predict(X_train)

            # make a prediction using the test data
            test_preds = cb_model.predict(X_test)

            # get train data R2 and MAE
            train_r2 = r2_score(y_train, train_preds)
            train_mae = mean_absolute_error(y_train, train_preds)
            train_r2s.append(train_r2)
            train_maes.append(train_mae)

            # get test data R2 and MAE
            test_r2 = r2_score(y_test, test_preds)
            test_mae = mean_absolute_error(y_test, test_preds)
            test_r2s.append(test_r2)
            test_maes.append(test_mae)

            # add test true vs predict to output
            true_arrays.append(y_test)
            predicted_arrays.append(test_preds)

        # get performance outputs into the performance log
        performance_dict = self.get_model_outputs(
            train_r2s,
            test_r2s,
            train_maes,
            test_maes,
        )

        self.performance_log.update({self.run: performance_dict})

        # get the true and predicted values into
        true_vs_predicted = {
            'true': np.concatenate(true_arrays),
            'predicted': np.concatenate(predicted_arrays),
        }
        self.true_vs_predicted.update({self.run: true_vs_predicted})

    def train_regressor(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        target: Union[np.ndarray, pd.Series],
        init_model: CatBoostRegressor = None,
        regressor_params: dict = None,
        k_fold_evaluation: bool = True,
        evaluation_kfolds: int = 10,
        verbose: bool = False,
    ) -> CatBoostRegressor:
        self.run += 1
        # prep the input data
        features, target = prep_training_inputs(features, target)

        # get a list of category features
        cat_features = list(features.select_dtypes('category').columns)

        # if we pass in a trained model, verify shapes match and continue training
        if isinstance(init_model, CatBoostRegressor):
            if self._verify_shapes(features, target):
                print('Continuing the training of any existing CatBoostRegressor model.'
                      ' Ignore if expected behavior w/ param:init_model.')
                init_model.fit(
                    features,
                    target,
                    cat_features=cat_features,
                    verbose=verbose,
                )
                return CatBoostRegressor
            else:
                return print(f'ERROR: The shape of param:features and/or param:target do'
                             'not match the supplied param:init_model. Initiating a new model.')

        # prep evaluation and hyper params
        if regressor_params is not None:
            regressor_params = self.clean_model_params(
                model_type='regressor',
                params_dict=regressor_params,
            )
            self.parameter_log.update({self.run: regressor_params})

        # train a model over a 10-Kfold CV
        if k_fold_evaluation:
            print(f'{evaluation_kfolds} K-Fold performance evaluation...')
            performance_dict = self.regression_kfold_evaluation(
                features,
                target,
                regressor_params,
                cat_features=cat_features,
                k_folds=evaluation_kfolds,
            )
            print(
                f'Done! {evaluation_kfolds} K-Fold perfomance metrics: {performance_dict}')

        # make final model and fit to the complete training dataset
        cb_model = cb_model = cb.to_regressor(
            cb.CatBoost(params=regressor_params))
        return cb_model.fit(
            features,
            target,
            cat_features=cat_features,
            verbose=verbose,
        )

    def train_classifier(self,
                         features: pd.DataFrame, target: pd.Series) -> object:
        pass
