"""
Agentpy Sampling Module
Content: Sampling functions
"""

import itertools
import numpy as np

from SALib.sample import saltelli
from .tools import param_tuples_to_salib


def sample(parameter_ranges, n):
    """ Creates a sample of different parameter combinations
    by seperating each range into 'n' values, using :func:`numpy.arange`.

    Arguments:
        parameter_ranges (dict): Dictionary of parameters.
            Only values that are given as a tuple will be varied.
            Tuple must be of style (min_value, max_value).
        n: Number of values to sample per varied parameter.

    See also:
        :func:`sample_discrete`, :func:`sample_saltelli`

    Returns:
        list of dict: List of parameter dictionaries
    """

    for k, v in parameter_ranges.items():
        if isinstance(v, tuple):
            parameter_ranges[k] = np.arange(v[0], v[1], (v[1] - v[0]) / n)
            if len(v) > 2:  # TODO Generalize
                parameter_ranges[k] = [v[2](i) for i in parameter_ranges[k]]
        else:
            parameter_ranges[k] = [v]

    param_combinations = list(itertools.product(*parameter_ranges.values()))
    param_sample = [{k: v for k, v in zip(parameter_ranges.keys(), parameters)}
                    for parameters in param_combinations]

    return param_sample


def sample_discrete(parameter_ranges):
    """ Creates a sample of different parameter combinations from all possible
    combinations within the passed parameter ranges.

    Arguments:
        parameter_ranges (dict): Dictionary of parameters.
            Only values that are given as a tuple will be varied.
            Tuple must be of style (value1, value2, value3, ...).

    See also:
        :func:`sample`, :func:`sample_saltelli`

    Returns:
        list of dict: List of parameter dictionaries
    """

    def make_tuple(v):
        if isinstance(v, tuple):
            return v
        else:
            return (v,)

    param_ranges_values = [make_tuple(v) for k, v in parameter_ranges.items()]
    param_combinations = list(itertools.product(*param_ranges_values))
    param_sample = [{k: v for k, v in zip(parameter_ranges.keys(), parameters)}
                    for parameters in param_combinations]

    return param_sample


def sample_saltelli(parameter_ranges, N, calc_second_order=True):  # noqa
    """ Creates a sample of different parameter combinations,
    using :func:`SALib.sample.saltelli.sample`.

    Arguments:
        parameter_ranges (dict): Dictionary of parameters.
            Only values that are given as a tuple will be varied.
            Tuple must be of style (min_value, max_value).
        N (int): The number of samples to generate,
            see :func:`SALib.sample.saltelli.sample`.
        calc_second_order (bool): Calculate second-order sensitivities,
            see :func:`SALib.sample.saltelli.sample` (default True).

    See also:
        :func:`sample`, :func:`sample_discrete`

    Returns:
        list of dict: List of parameter dictionaries
    """

    # STEP 1 - Convert param_ranges to SALib Format
    param_ranges_tuples = {k: v for k, v in parameter_ranges.items()
                           if isinstance(v, tuple)}
    param_ranges_salib = param_tuples_to_salib(param_ranges_tuples)

    # STEP 2 - Create SALib Sample
    salib_sample = saltelli.sample(param_ranges_salib, N, calc_second_order)

    # STEP 3 - Convert back to Agentpy Parameter Dict List
    ap_sample = []

    for param_instance in salib_sample:

        parameters = {}
        parameters.update(parameter_ranges)

        for i, key in enumerate(param_ranges_tuples.keys()):
            parameters[key] = param_instance[i]

        ap_sample.append(parameters)

    return ap_sample
