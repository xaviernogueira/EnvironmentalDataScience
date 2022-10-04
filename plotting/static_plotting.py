import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score


# define a function to control subplots grid creation - IGNORE, used within other functions.
def make_subplot_grid(n: int, figsize: tuple = (10, 8), max_cols: int = 3) -> tuple:
    """
    Creates a matplotlib Figure, Axes grid for plotting.
    :param n: (int) number of plots to make.
    :param figsize: (tuple) a figsize tuple to control subplot creation, default=(18, 10).
    :param max_cols: (int) maximum # of columns to fit plots on, default=5.
    :returns: (list) storing Figure, Axes objects (i.e., fig, axes = plt.subplots()).
    """
    # if < 3 variables plot them inline. If exactly 4 plot a square.
    if n <= 3:
        rows, cols = 1, n
    elif n == 4:
        rows, cols = 2, 2
    else:
        # iterate over divisibility to find optimal plotting grid
        for i in [i for i in ([3, 4, 2, 5] + list(range(5, max_cols))) if i <= max_cols]:
            if n % i == 0:
                return plt.subplots(int(n / i), i, figsize=figsize)

        # if indivisible just make columns of 3 and have an hideous one at the bottom
        rows = int(n // 3) + 1
        cols = 3

    # get optimzal figsize if not hardcoded
    if figsize is None:
        figsize = (int(3.5 * cols), int(2 * rows))
    return plt.subplots(rows, cols, figsize=figsize)


# define a function to plot histograms for N variables in a subplot grid
def plot_histograms(df: pd.DataFrame, columns: list = None,
                   figsize: tuple = None, max_cols: int = 3) -> matplotlib.figure.Figure:
    """
    Uses seaborn on matplotlib subplots to plot a grid of histograms plots for N variables
    """
    # only use columns that are in df, make the subplot grid
    if columns is not None:
        columns = [i for i in columns if i in list(df.columns)]
    else:
        columns = list(df.columns)
    print(f'Plotting the following columns: {columns}')
    fig, axes = make_subplot_grid(n=len(columns), figsize=figsize, max_cols=max_cols)

    # if just one variable, plot it
    if len(columns) == 1:
        header = columns[0]
        sns.histplot(ax=axes, data=df, x=header)

    # plot if there is only one row
    elif axes.shape[0] == 1:
        for count in range(len(columns)):
            header = columns[count]
            sns.histplot(ax=axes[count], data=df, x=header)

    # otherwise populate the axes grid
    else:
        count = 0
        print(axes.shape)
        while count < len(columns):
            # iterate over rows
            for i in range(axes.shape[0]):
                # iterate over columns
                for j in range(axes.shape[-1]):
                    try:
                        header = columns[count]
                        axes[i, j] = sns.histplot(ax=axes[i, j], data=df, x=header,
                                                  stat='proportion', kde=True)
                        axes[i, j].set_xlim(0, None)
                        count += 1
                    except IndexError:
                        pass
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.3, hspace=0.8)
    plt.tight_layout()


# define a pie plot function for categorical variables
def plot_piecharts(df: pd.DataFrame, bool_columns: list) -> matplotlib.figure.Figure:
    """
    Uses seaborn on matplotlib subplots to plot a grid of pie plots for N boolean dtype variables.
    """
    # only use columns that are in df, make the subplot grid
    columns = [i for i in bool_columns if i in list(df.columns)]
    fig, axes = make_subplot_grid(n=len(columns))

    # add data to subplots and label appropriately
    prop_dict = {}
    labels_dict = {}

    df_length = df.shape[0]
    for bool_col in columns:
        num_na = df.loc[df[bool_col].isna()].shape[0]
        prop_true = round(float(df.loc[df[bool_col] == True].shape[0] / (df_length - num_na)))
        prop_false = round(float(1 - prop_true))

        # set labels for categories
        _labels = ['Yes/True', 'No/False', 'NoData']

        # add category sums to prop_dict with column headers as keys
        props = [prop_true, prop_false, round(float(num_na / df_length), 2)]
        outlist = []
        labels = []
        for i, prop in enumerate(props):
            print(f'{i} - {prop} - {_labels[i]}')
            if prop > 0:
                outlist.append(prop)
                labels.append(_labels[i])
        prop_dict[bool_col] = outlist
        labels_dict[bool_col] = labels

    # if just one variable, plot it
    if len(columns) == 1:
        header = columns[0]
        sizes = prop_dict[header]
        bool_col = columns[0]
        axes.pie(sizes, labels=labels_dict[bool_col], autopct='%1.1f%%')
        axes.set_title(bool_col)

    # plot if there is only one row
    elif axes.shape[0] == 1:
        for count in range(len(columns)):
            header = columns[count]
            sizes = prop_dict[header]
            bool_col = columns[count]
            axes[count].pie(sizes, labels=labels_dict[bool_col], autopct='%1.1f%%')
            axes[count].set_title(bool_col)

    # otherwise, populate the axes grid
    else:
        count = 0
        while count < len(columns):
            # iterate over rows
            for i in range(axes.shape[0]):
                # iterate over columns
                for j in range(axes.shape[-1]):
                    header = columns[count]
                    sizes = prop_dict[header]
                    bool_col = columns[count]
                    axes[i, j].pie(sizes, labels=labels_dict[bool_col], autopct='%1.1f%%')
                    axes[i, j].set_title(bool_col)
                    count += 1
    plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.3, hspace=0.5)


# define a function to plot linear fit
def _linearfit_plot(df: pd.DataFrame, x_col: str, y_col: str, model: LinearRegression) -> matplotlib.figure.Figure:
    """
    :param df: (pd.DataFrame) a dataframe containing both param:x_col and param:y_col.
    :param x_col: (str) x column header (independent variable, i.e., income).
    :param y_col: (str) y column header (dependent variable, i.e., high school graduation rate).
    :param linear_coefs: (tuple) the [1] output from pyfunc:x_variable_normalize() -> (coef, intercept)
    """
    df = df.dropna(subset=[x_col, y_col], axis=0, how='any')
    fig, ax = plt.subplots()
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # calculate point density for better visualization
    xy = np.vstack([x, y])
    c = gaussian_kde(xy)(xy)

    # get the linear preadiction line and r2 score
    predicted_y = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y, predicted_y)

    # plot the scatter plot with density shading
    plt.grid(which='major', color='lightgray', linestyle='--')
    ax.scatter(x, y, c=c, s=20)

    # plot the linear fit
    _x = np.arange(0, x.max(), 10).reshape(-1, 1)
    _y = model.predict(_x)
    plt.plot(_x, _y, color='red', label='linear_fit')

    # set axes maximums to mean + 3xSTD to avoid outliers skewing plot
    plt.xlim(0, np.mean(x) + (3 * np.std(x)))
    plt.ylim(0, np.mean(y) + (3 * np.std(y)))
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Linear Regression Plot')
    plt.annotate('R2 = %s' % round(r2, 2), (0.15, 0.8), xycoords='subfigure fraction', fontsize='large')
    plt.tight_layout()
    pass