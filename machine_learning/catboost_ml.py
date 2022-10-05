from catboost import CatBoostRegressor, CatBoostClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import seaborn as sns
from scipy.stats import gaussian_kde
import sklearn as skl
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

# import type hints
from typing import Union, Tuple, List

# import protocol to implement
from protocols import MachineLearningModel

# functions to move to preprocessing_functions.py


def prep_training_inputs(features, target,
                         allow_index_join: bool = True) -> Tuple(pd.DataFrame, pd.Series):
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


def clean_model_params(model_instance: object, hyperparams: dict) -> dict:
    """
    Returns the hyperparams dictionary removing any hyperparams kwargs not
    applicable to CatBoost.
    """
    pass


class CatBoostML(MachineLearningModel):
    # CatBoost specific extension functions

    # MachineLearningModel protocol function implementations
    def train_regressor(self,
                        features: Union[np.ndarray, pd.DataFrame],
                        target: Union[np.ndarray, pd.Series],
                        hyper_params: dict = None,
                        evaluation_params: dict = None,
                        evaluation_kfolds: int = 10,
                        **kwargs) -> Tuple(object, dict):

        # prep the input data
        features, target = prep_training_inputs(features, target)

        # train a model over a 10-Kfold CV
        kfolds = skl.model_selection.KFold(n_splits=evaluation_kfolds)

        # make two list to store true and predicted values (in order to make performance charts)
        true_arrays = []
        predicted_arrays = []

        for train_index, test_index in kfolds.split(features):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = features.iloc[train_index], features.iloc[test_index]

            # build regression or classification model (uses Logloss optimization for binary classification data and Accuracy eval_metric)
            cb_model = CatBoostRegressor(eval_metric='R2',
                                         loss_function='RMSE',
                                         od_type='Iter',
                                         od_wait=250)

            cb_model = cb_model.fit(X_train, y_train, verbose=False)

            # make a prediction using the training data
            train_preds = cb_model.predict(X_train)

            # make a prediction using the test data
            test_preds = cb_model.predict(X_test)

            # get train data R2 and MAE
            train_r2 = r2_score(y_train, train_preds)
            train_mae = skl.metrics.mean_absolute_error(y_train, train_preds)
            train_r2s.append(train_r2)
            train_maes.append(train_mae)

            # get test data R2 and MAE
            test_preds = cb_model.predict(X_test)
            test_r2 = r2_score(y_test, test_preds)
            test_mae = skl.metrics.mean_absolute_error(y_test, test_preds)
            test_r2s.append(test_r2)
            test_maes.append(test_mae)

        # make final model
        cb_model = CatBoostRegressor(
            eval_metric='R2', loss_function='RMSE', od_type='Iter', od_wait=250)

        # get mean and STD values for R2 and MAE using K-fold CV data
        out_list = get_model_outputs(depend_col,
                                     train_r2s,
                                     test_r2s,
                                     train_maes,
                                     test_maes)

        # fit output model to the complete training dataset
        cb_model = cb_model.fit(features, target, verbose=False)

        # convert lists of true and predicted values to arrays
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)
        true_predict_values = [true_values, predicted_values]
        pass

    def train_classifier(self,
                         features: pd.DataFrame, target: pd.Series) -> object:
        pass
