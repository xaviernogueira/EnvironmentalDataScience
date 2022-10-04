from typing import Union
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from random import randint


def sankey_plot(in_df: pd.DataFrame, ordered_col_list: list, ignore_states: Union[list, str] = None, out_dir=None,
                out_name: str = None):
    """
    Plots the volume of transitions between discrete states
    :param in_df: a pandas dataframe with columns representing object states that are transitioned between.
    :param ordered_col_list: a list with the column headers of the [first_state, second_state, ..., n_state].
    :param ignore_states: a list with any unique column values that should be excluded from the plot.
    :param out_dir: (optional) string path to which the plot is saved, otherwise it is displayed in browser.
    :param out_name: (optional) string name w/o extension for the file. Default is sankey_plot.
    :return: an interactive Plotly plot in a browser window
    """
    # initialize plotting lists
    source = []
    target = []
    value = []
    transitions = []
    unique_values = []

    # get values on both sides of each transition
    for i, col in enumerate(ordered_col_list[:-1]):
        col1 = in_df[col].to_list()
        col2 = in_df[ordered_col_list[i + 1]].to_list()
        transition = list(zip(col1, col2))
        transitions.append(transition)

        # get unique state values
        for _col in [col1, col2]:
            for val in set(col1):
                if val not in unique_values:
                    unique_values.append(val)

    # get unique transitions, and initialize a list to store counts for each transitions
    unique_transitions = []
    unique_counts = []
    unique_values.sort()

    for transition in transitions:
        _unique_transitions1 = list(set(transition))

        # if ignore states are specified, remove those transitions from the analysis
        _unique_transitions = []
        if ignore_states is not None:
            for t in _unique_transitions1:
                t1, t2 = t  # t1 and t2 are unpacked from the tuple
                if t1 not in ignore_states and t2 not in ignore_states:
                    _unique_transitions.append(t)
        else:
            _unique_transitions = _unique_transitions1

        _unique_counts = list(np.zeros(len(_unique_transitions), dtype=int))
        unique_transitions.append(_unique_transitions)
        unique_counts.append(_unique_counts)

    # make a list storing each unique state transition and the number of occurrences for each transition
    transition_counts = []
    for j, all_zipped in enumerate(transitions):
        _unique_transitions = unique_transitions[j]
        _unique_counts = unique_counts[j]
        for pair in all_zipped:
            try:
                i = _unique_transitions.index(pair)
                _unique_counts[i] += 1
            except ValueError:
                pass

        # zip the unique transition and the number of occurrences, add to transition_counts list
        _transition_counts = list(zip(_unique_transitions, _unique_counts))
        transition_counts.append(_transition_counts)

    # remove ignore states from the plot
    if ignore_states is not None:
        unique_values = [i for i in unique_values if i not in ignore_states]
    max_unique_values = len(unique_values)

    # get colors for each category
    color_scheme = []
    for i in range(max_unique_values):
        color_scheme.append('#%06X' % randint(0, 0xFFFFFF))

    # get x and y position for nodes
    x_gap = 1 / len(ordered_col_list)
    y_gap = 1 / max_unique_values

    labels = []
    x_locs = []
    y_locs = []
    colors = []

    for u, col in enumerate(ordered_col_list):
        x_loc = (0.1 + (0.9 * (x_gap * u)))
        for i, unique in enumerate(unique_values):
            labels.append(unique)
            x_locs.append(x_loc)
            y_locs.append(0.2 + (0.7 * (y_gap * i)))
            colors.append(color_scheme[i])

    # format sankey nodes dictionary
    nodes = {
        "label": labels,
        "x": y_locs,
        "color": colors,
        'pad': 15}

    # iterate over all unique transitions and their counts, use this to fill in source-target-value lists
    for t_index, transition_count_list in enumerate(transition_counts):
        spacer = t_index * max_unique_values
        for transition_tuple in transition_count_list:
            state_1, state_2 = transition_tuple[0]
            count = transition_tuple[1]

            # find index in non-repeated labels list for both sides of the transition
            base_index_1 = unique_values.index(state_1)
            base_index_2 = unique_values.index(state_2)

            # add to source, target, values lists via labels list indices
            state_1_index = base_index_1 + spacer
            state_2_index = base_index_2 + spacer + max_unique_values
            source.append(state_1_index)
            target.append(state_2_index)
            value.append(count)

    # plot the Sankey (each transition has a source node, target node, and weight)
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=nodes,  # 10 Pixels
        link={
            "source": source,
            "target": target,
            "value": value}))

    # save plot if desired
    if out_dir is not None:
        if out_name is not None:
            out_fig = out_dir + f'\\{out_name}.png'
        else:
            out_fig = out_dir + '\\sankey_plot.png'
        fig.write_image(out_fig, scale=5)
        print(f'Sankey plots saved @ {out_fig}')

    return fig.show()