import pandas as pd
import numpy as np


def sankey_test_data(n_of_states: int = 3, n_of_classes: int = 5, rows: int = 1000):
    """
    Creates test data for plotting/interactive_plotting.py:sankey_plot().
    :param n_of_states: (int) number of unique time-steps / states to model transitions between.
    :param n_of_classes: (int) number of unique classes to a row can belong to.
    :param rows: (int) number of data rows.
    :return: (list[pd.DataFrame, list]) the output dataframe [0], and a list of the state columns [1].
    """
    test_df = pd.DataFrame()
    test_ids = range(0, rows)
    test_df['Item_ID'] = np.array(test_ids)

    classes = np.arange(n_of_classes)
    ordered_col_list = []
    for state in range(0, n_of_states):
        random_classes = []
        for i in test_ids:
            sample = np.random.choice(classes)
            random_classes.append(sample)
        col_name = f'state{state}'
        test_df[col_name] = np.array(random_classes)
        ordered_col_list.append(col_name)

    return test_df, ordered_col_list
