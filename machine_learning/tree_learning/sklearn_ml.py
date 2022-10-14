import sklearn as skl
def make_rf_model(train_df, depend_col, classification=False):
    """
    Trains a Random Forest model to predict the depend_col variable and applied 10-K fold CV to calculate performance metrics.
    :param train_df: the independent variable portion of knn_df (DataFrame)
    :param depend_col: the dependent variable column header (str)
    :param classification: if False (default), a Regression model is built with a Sqaured Error loss function. If True, a Classifier model is build with Gini criterion.
    :returns: a list containing [the RandomForest model instance, a list of performance metrics, a list containing an array of true values [0] and predicted values [1]]
    """
    # make training set from only not na rows in the dependent variabe
    Y_train_full = train_df[depend_col]
    X_train_full = train_df.drop(depend_col, axis=1)

    # make lists to hold k-fold result values for Regression
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

        # build regression or classification model
        if not classification:
            rf_model = RandomForestRegressor(max_depth=5, criterion='squared_error')
        else:
            rf_model = RandomForestClassifier(criterion='gini', max_depth=5)
            y_train = y_train.astype('int')
            y_test = y_test.astype('int')
        rf_model = rf_model.fit(X_train, y_train)

        # make a prediction using the training data
        train_preds = rf_model.predict(X_train)

        # make a prediction using the test data
        test_preds = rf_model.predict(X_test)

        # calculate Regression or Classifier performance metrics
        if not classification:
            # get train data R2 and MAE
            train_r2 = r2_score(y_train, train_preds)
            train_mae = skl.metrics.mean_absolute_error(y_train, train_preds)
            train_r2s.append(train_r2)
            train_maes.append(train_mae)

            # get test data R2 and MAE
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
        rf_model = RandomForestRegressor(max_depth=5, criterion='squared_error')

        # get mean and STD values for R2 and MAE using K-fold CV data
        out_list = get_model_outputs(depend_col, train_r2s, test_r2s, train_maes, test_maes)
    else:
        Y_train_full = Y_train_full.astype('int')
        rf_model = RandomForestClassifier(criterion='gini', max_depth=5)

        # get classification performance metrics
        out_list = get_class_model_outputs(depend_col, train_accuracies, train_f1_scores, test_accuracies, test_f1_scores)

    rf_model = rf_model.fit(X_train_full, Y_train_full)

    # convert lists of true and predicted values to arrays
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    true_predict_values = [true_values, predicted_values]

    return rf_model, out_list, true_predict_values
