import catboost as cb
from catboost import CatBoostRegressor, CatBoostClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import sklearn as skl
from sklearn.metrics import r2_score


# ################ V1 FUNCTIONS - OUTDATED AND BAD IMO ####################
def get_model_outputs(depend_col, train_r2s, test_r2s, train_maes, test_maes):
    """
    A function that returns a list containing mean and STD R2 and MAE values from 10-K fold CV for a given dependent variable's Regression model.
    :returns: a list containing [depend_col (str), mean_train_r2, mean_test_r2, std_train_r2, std_test_r2, mean_train_mae, mean_test_mae, std_train_mae, std_test_mae]
    """
    # convert train data to arrays
    np_train_r2s = np.array(train_r2s)
    np_train_maes = np.array(train_maes)

    # convert test data to arrays
    np_test_r2s = np.array(test_r2s)
    np_test_maes = np.array(test_maes)

    # create out_list with model performance metrics
    mean_train_r2 = np_train_r2s.mean()
    std_train_r2 = np_train_r2s.std()

    mean_train_mae = np_train_maes.mean()
    std_train_mae = np_train_maes.std()

    mean_test_r2 = np_test_r2s.mean()
    std_test_r2 = np_test_r2s.std()

    mean_test_mae = np_test_maes.mean()
    std_test_mae = np_test_maes.std()

    out_list = [depend_col, mean_train_r2, mean_test_r2, std_train_r2,
                std_test_r2, mean_train_mae, mean_test_mae, std_train_mae, std_test_mae]
    return out_list


def get_class_model_outputs(depend_col, train_accuracies, train_f1_scores, test_accuracies, test_f1_scores):
    """
    A function that calculates the mean and STD Classifier performance scores, accuracy and F1-Score from 10-K fold CV for a given dependent variable's Classifier model.
    :returns: a list containing [depend_col (str), mean_train_accuracy, mean_test_accuracy, std_train_accuracy, std_test_accuracy, \
    mean_train_f1_score, mean_test_f1_score, std_train_f1_score, std_test_f1_score]
    """
    # convert train data to arrays
    np_train_accuracies = np.array(train_accuracies)
    np_train_f1_scores = np.array(train_f1_scores)

    # convert test data to arrays
    np_test_accuracies = np.array(test_accuracies)
    np_test_f1_scores = np.array(test_f1_scores)

    # create out_list with model performance metrics
    mean_train_accuracy = np_train_accuracies.mean()
    std_train_accuracy = np_train_accuracies.std()

    mean_train_f1_score = np_train_f1_scores.mean()
    std_train_f1_score = np_train_f1_scores.std()

    mean_test_accuracy = np_test_accuracies.mean()
    std_test_accuracy = np_test_accuracies.std()

    mean_test_f1_score = np_test_f1_scores.mean()
    std_test_f1_score = np_test_f1_scores.std()

    # create out_list
    out_list = [depend_col, mean_train_accuracy, mean_test_accuracy, std_train_accuracy, std_test_accuracy,
                mean_train_f1_score, mean_test_f1_score, std_train_f1_score, std_test_f1_score]

    return out_list


def make_catboost_model(train_df, depend_col, classification=False):
    """Trains a CatBoost Regression model to predict the depend_col variable and applied 10-K fold CV to calculate performance metrics.
    We use the 'Iter' overfit protection method cutting off at training at 500 iterations. The loss function is RMSE.
    :param train_df: the independent variable portion of knn_df (DataFrame)
    :param depend_col: the dependent variable column header (str)
    :param classification: if False (default), a Regression model is built with a RMSE loss function. If True, a Classifier model is built using LogLoss.
    :returns: a list containing [the CatBoost model instance, a list of performance metrics, a list containing an array of true values [0] and predicted values [1]]
    """
    # make training set from only not na rows in the dependent variabe
    Y_train_full = train_df[depend_col]
    X_train_full = train_df.drop(depend_col, axis=1)

    # make lists to hold k-fold result values
    train_r2s = []
    train_maes = []
    test_r2s = []
    test_maes = []

    # make lists to hold k-fold result values for Classification
    train_accuracies = []
    train_f1_scores = []
    test_accuracies = []
    test_f1_scores = []

    # train a model over a 10-Kfold CV
    kfolds = skl.model_selection.KFold(n_splits=10)

    # make two list to store true and predicted values (in order to make performance charts)
    true_values = []
    predicted_values = []

    for train_index, test_index in kfolds.split(X_train_full):
        X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]
        y_train, y_test = Y_train_full.iloc[train_index], Y_train_full.iloc[test_index]

        # build regression or classification model (uses Logloss optimization for binary classification data and Accuracy eval_metric)
        if not classification:
            cb_model = CatBoostRegressor(eval_metric='R2', loss_function='RMSE', od_type='Iter', od_wait=250)
        else:
            cb_model = CatBoostClassifier(eval_metric='Accuracy', max_depth=5, od_type='Iter', od_wait=250)
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')

        cb_model = cb_model.fit(X_train, y_train, verbose=False)

        # make a prediction using the training data
        train_preds = cb_model.predict(X_train)

        # make a prediction using the test data
        test_preds = cb_model.predict(X_test)

        if not classification:
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

        else:
            # convert to int
            train_preds = train_preds.astype('int')
            test_preds = test_preds.astype('int')

            # calculate train accuracy and F1 score
            train_accuracy = skl.metrics.accuracy_score(y_train, train_preds)
            train_f1_score = skl.metrics.f1_score(y_train, train_preds)
            train_accuracies.append(train_accuracy)
            train_f1_scores.append(train_f1_score)

            # calculate test accuracy and F1 score
            test_accuracy = skl.metrics.accuracy_score(y_test, test_preds)
            test_f1_score = skl.metrics.f1_score(y_test, test_preds)
            test_accuracies.append(test_accuracy)
            test_f1_scores.append(test_f1_score)

        # add y_test and test_preds to lists for output
        true_values.extend(list(y_test))
        predicted_values.extend(list(test_preds))

    # make final model
    if not classification:
        cb_model = CatBoostRegressor(eval_metric='R2', loss_function='RMSE', od_type='Iter', od_wait=250)

        # get mean and STD values for R2 and MAE using K-fold CV data
        out_list = get_model_outputs(depend_col, train_r2s, test_r2s, train_maes, test_maes)
    else:
        Y_train_full = Y_train_full.astype('int')
        cb_model = CatBoostClassifier(eval_metric='Accuracy', max_depth=5, od_type='Iter', od_wait=250)

        # get classification performance metrics
        out_list = get_class_model_outputs(depend_col, train_accuracies, train_f1_scores, test_accuracies, test_f1_scores)

    # fit output model to the complete training dataset
    cb_model = cb_model.fit(X_train_full, Y_train_full, verbose=False)

    # convert lists of true and predicted values to arrays
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    true_predict_values = [true_values, predicted_values]

    return cb_model, out_list, true_predict_values


def impute_values(metrics_df: pd.DataFrame, model: cb.CatBoostRegressor, knn_df: pd.DataFrame,
                  depend_col: str) -> pd.DataFrame:
    """
    This function applies the trained ML regressor object to impute the missing values
    in the param:depend_col column of param:metrics_df, using param:knn_df as the training data.
    """
    # split test data into independent and dependent variable DataFrames
    X_test = knn_df.drop(columns=[depend_col], axis=1)

    # make prediction for missing values
    imputed_values = model.predict(X_test)
    metrics_df.loc[metrics_df[depend_col].isna()] = imputed_values.loc[metrics_df[depend_col].isna()]

    # get list of non-imputed values
    real_list = metrics_df.loc[metrics_df[depend_col].notna()].tolist()

    return metrics_df


def show_true_vs_predicted(variable_name, performance_df, true_vs_predict_dict, model_name=None):
    """
    Makes a scatter plot with a point density heatmap to show true vs predicted values for a given variable
    :param variable_name: column header name (str)
    :param performance_df: the performance DataFrame storing K-fold values
    :param true_vs_predict_dict: the dictionary storing true and predicte values with variable names as keys
    :param model_name: name of the model (str) for plot labeling
    """
    # get R2 values from the performance dataframe
    r2 = performance_df.set_index('variable').at[variable_name, 'mean_test_r2']

    # get true and predicted values from the appropriate true_vs_predict_dict
    y_test, prediction = true_vs_predict_dict[variable_name]

    # Calculate the point density
    xy = np.vstack([y_test, prediction])
    z = gaussian_kde(xy)(xy)

    # make the plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, prediction, c=z, s=20)

    # have option to format if user forgets to input the model name
    if not model_name:
        model_name = 'Unknown model'

    # format the plot
    plt.title(f'{model_name} Regression - Predicting {variable_name}')
    plt.plot(np.arange(0, 1000, 0.1), np.arange(0, 1000, 0.1), c='red')
    plt.xlim(0, np.max(y_test))
    plt.ylim(0, np.max(prediction))
    plt.xlabel(f'Actual {variable_name}')
    plt.ylabel(f'Predicted {variable_name}')
    plt.annotate('R2 = %s' % round(r2, 2), (0.15, 0.8), xycoords='subfigure fraction', fontsize='large')

    return plt.show()


def show_accuracy_confusion_matrix(variable_name, true_vs_predict_dict, model_name=None):
    """
    This function plots an accuracy confusion matrix for a given boolean variable (i.e., imputed via a Classifier algo)
    :param variable_name: column header name (str)
    :param true_vs_predict_dict: the dictionary storing true and predicte values with variable names as keys
    :param model_name: name of the model (str) for plot labeling
    """
    # get true and predicted values from the appropriate true_vs_predict_dict
    true, predict = true_vs_predict_dict[variable_name]

    # create confusion matrix numpy array
    tn, fp, fn, tp = skl.metrics.confusion_matrix(true, predict, normalize='all').ravel()
    confusion_matrix = np.array([(tp, fp), (fn, tn)])

    # plot the confusion matrix as a heatplot
    confuse_labels = ['Positive', 'Negative']
    f, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix, cmap="YlGnBu", vmin=0, vmax=1, annot=True, xticklabels=confuse_labels, yticklabels=confuse_labels)

    # have option to format if user forgets to input the model name
    if not model_name:
        model_name = 'Unknown model'

    # format the plot
    plt.title(f'{model_name} Classifier - Predicting {variable_name}')
    plt.tick_params(axis='both', which='major', labelsize=14, labelbottom= False, top=True, labeltop=True)
    plt.xlabel('TRUE', fontsize=16)
    plt.ylabel('PREDICTED', fontsize=16)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()

    return plt.show()