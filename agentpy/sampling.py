"""
Agentpy Sampling Module
Content: Sampling functions
"""

import itertools
import numpy as np

from SALib.sample import saltelli
from .tools import param_tuples_to_salib

def sample(parameter_ranges, n, digits=None):
    """ Creates a sample of different parameter combinations
    by seperating each range into 'n' values, using :func:`numpy.linspace`.

    Arguments:
        parameter_ranges(dict): Dictionary of parameters.
            Only values that are given as a tuple will be varied.
            Tuple must be of the following style: (min_value, max_value).
            If both values are of type int,
            the output will be rounded and converted to int.
        n(int): Number of values to sample per varied parameter.
        digits(int, optional):
            Number of digits to round the output values to (default None).

    Returns:
        list of dict: List of parameter dictionaries
    """

    parameter_ranges = dict(parameter_ranges)
    for k, v in parameter_ranges.items():
        if isinstance(v, tuple):
            p_range = np.linspace(v[0], v[1], n)
            if all([isinstance(pv, int) for pv in v[:2]]):
                p_range = [int(round(pv)) for pv in p_range]
            elif digits is not None:
                p_range = [round(pv, digits) for pv in p_range]
            parameter_ranges[k] = p_range
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
            Tuples must be of the following style:
            (value1, value2, value3, ...).

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


def _is_int(param_tuples):
    is_int = {}
    for key, p_values in param_tuples.items():
        is_int[key] = all([isinstance(v, int) for v in p_values[:2]])
    return is_int


def sample_saltelli(parameter_ranges, n, calc_second_order=False, digits=None):
    """ Creates a sample of different parameter combinations,
    using :func:`SALib.sample.saltelli.sample`. This sample can later be used
    to calculate Sobol sensitivity indices with :func:`sensitivity_sobol`.

    Arguments:
        parameter_ranges (dict): Dictionary of parameters.
            Only values that are given as a tuple will be varied.
            Tuple must be of the following style: (min_value, max_value).
            If both values are of type int,
            the output will be rounded and converted to int.
        n (int): The number of samples to generate,
            see :func:`SALib.sample.saltelli.sample`.
        calc_second_order (bool, optional):
            Create sample that can be used by :func:`sensitivity_sobol`
            to calculate second-order sensitivities (default False).
        digits (int, optional):
            Number of digits to round the output values to (default None).

    Returns:
        list of dict: List of parameter dictionaries
    """

    # STEP 0 - Find variable parameters and check type
    param_ranges_tuples = {k: v for k, v in parameter_ranges.items()
                           if isinstance(v, tuple)}

    # STEP 1 - Convert param_ranges to SALib Format
    param_ranges_salib = param_tuples_to_salib(param_ranges_tuples)

    # STEP 2 - Create SALib Sample
    salib_sample = saltelli.sample(param_ranges_salib, n, calc_second_order)

    # STEP 3 - Convert back to Agentpy Parameter Dict List and adjust values
    ap_sample = []
    is_int = _is_int(param_ranges_tuples)

    for param_instance in salib_sample:

        parameters = {}
        parameters.update(parameter_ranges)

        for i, key in enumerate(param_ranges_tuples.keys()):
            p = param_instance[i]
            if is_int[key]:  # Convert to integer
                p = int(round(p))
            elif digits:  # Round to digits
                p = round(p, digits)
            parameters[key] = p

        ap_sample.append(parameters)

    return ap_sample
