"""
Agentpy Sampling Module
Content: Sampling functions
"""

# TODO Latin Hypercube
# TODO Random distribution samples
# TODO Store meta-info for later analysis

import itertools
import random
import numpy as np

from SALib.sample import saltelli
from .tools import param_tuples_to_salib, InfoStr, AgentpyError


class Range:
    """ A range of parameter values
    that can be used to create a :class:`Sample`.

    Arguments:
        vmin (float, optional):
            Minimum value for this parameter (default 0).
        vmax (float, optional):
            Maximum value for this parameter (default 1).
        vdef (float, optional):
            Default value. Default value. If none is passed, `vmin` is used.
    """

    def __init__(self, vmin=0, vmax=1, vdef=None):
        self.vmin = vmin
        self.vmax = vmax
        self.vdef = vdef if vdef else vmin
        self.ints = False

    def __repr__(self):
        return f"Parameter range from {self.vmin} to {self.vmax}"


class IntRange(Range):
    """ A range of integer parameter values
    that can be used to create a :class:`Sample`.
    Similar to :class:`Range`,
    but sampled values will be rounded and converted to integer.

    Arguments:
        vmin (int, optional):
            Minimum value for this parameter (default 0).
        vmax (int, optional):
            Maximum value for this parameter (default 1).
        vdef (int, optional):
            Default value. If none is passed, `vmin` is used.
    """

    def __init__(self, vmin=0, vmax=1, vdef=None):
        self.vmin = int(round(vmin))
        self.vmax = int(round(vmax))
        self.vdef = int(round(vdef)) if vdef else vmin
        self.ints = True

    def __repr__(self):
        return f"Integer parameter range from {self.vmin} to {self.vmax}"


class Values:
    """ A pre-defined set of discrete parameter values
    that can be used to create a :class:`Sample`.

    Arguments:
        *args:
            Possible values for this parameter.
        vdef:
            Default value. If none is passed, the first passed value is used.
    """

    def __init__(self, *args, vdef=None):
        self.values = args
        self.vdef = vdef if vdef else args[0]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"Set of {len(self.values)} parameter values"


class Sample:
    """ A sequence of parameter combinations
    that can be used for :class:`Experiment`.

    Arguments:

        parameters (dict):
            Dictionary of parameter keys and values.
            Entries of type :class:`Range` and :class:`Values`
            will be sampled based on chosen `method` and `n`.
            Other types wil be interpreted as constants.

        n (int, optional):
            Sampling factor used by chosen `method` (default None).

        method (str, optional):
            Method to use to create parameter combinations
            from entries of type :class:`Range`. Options are:

            - ``linspace`` (default):
              Arange `n` evenly spaced values for each :class:`Range`
              and combine them with given :class:`Values` and constants.
              Additional keyword arguments:

                - ``product`` (bool, optional):
                  Return all possible combinations (default True).
                  If False, value sets are 'zipped' so that the i-th
                  parameter combination contains the i-th entry of each
                  value set. Requires all value sets to have the same length.

            - ``saltelli``:
              Apply Saltelli's sampling scheme,
              using :func:`SALib.sample.saltelli.sample` with `N=n`.
              This enables the analysis of Sobol Sensitivity Indices
              with :func:`DataDict.calc_sobol` after the experiment.
              Additional keyword arguments:

                - ``calc_second_order`` (bool, optional):
                  Whether to calculate second-order indices (default True).

        randomize (bool, optional):
            Whether to use the constant parameter 'seed' to generate different
            random seeds for every parameter combination (default True).
            If False, every parameter combination will have the same seed.
            If there is no constant parameter 'seed',
            this option has no effect.

        **kwargs: Additional keyword arguments for chosen `method`.

    """

    def __init__(self, parameters, n=None,
                 method='linspace', randomize=True, **kwargs):

        self._log = {'type': method, 'n': n, 'randomized': False}
        self._sample = getattr(self, f"_{method}")(parameters, n, **kwargs)
        if 'seed' in parameters and randomize:
            ranges = (Range, IntRange, Values)
            if not isinstance(parameters['seed'], ranges):
                seed = parameters['seed']
                self._log['randomized'] = True
                self._log['seed'] = seed
                self._assign_random_seeds(seed)

    def __repr__(self):
        return f"Sample of {len(self)} parameter combinations"

    def __iter__(self):
        return iter(self._sample)

    def __len__(self):
        return len(self._sample)

    # Sampling methods ------------------------------------------------------ #

    def _assign_random_seeds(self, seed):
        rng = random.Random(seed)
        for parameters in self._sample:
            parameters['seed'] = rng.getrandbits(128)

    @staticmethod
    def _linspace(parameters, n, product=True):

        params = {}
        for k, v in parameters.items():
            if isinstance(v, Range):
                if n is None:
                    raise AgentpyError(
                        "Argument 'n' must be defined for Sample "
                        "if there are parameters of type Range.")
                if v.ints:
                    p_range = np.linspace(v.vmin, v.vmax+1, n)
                    p_range = [int(pv)-1 if pv == v.vmax+1 else int(pv)
                               for pv in p_range]
                else:
                    p_range = np.linspace(v.vmin, v.vmax, n)
                params[k] = p_range
            elif isinstance(v, Values):
                params[k] = v.values
            else:
                params[k] = [v]

        if product:
            # All possible combinations
            combos = list(itertools.product(*params.values()))
            sample = [{k: v for k, v in zip(params.keys(), c)} for c in combos]
        else:
            # Parallel combinations (index by index)
            r = range(min([len(v) for v in params.values()]))
            sample = [{k: v[i] for k, v in params.items()} for i in r]

        return sample

    def _saltelli(self, params, n, calc_second_order=True):

        # STEP 0 - Find variable parameters and check type
        param_ranges_tuples = {}
        for k, v in params.items():
            if isinstance(v, Range):
                if v.ints:
                    # Integer conversion rounds down, +1 includes last integer
                    param_ranges_tuples[k] = (v.vmin, v.vmax+1)
                else:
                    param_ranges_tuples[k] = (v.vmin, v.vmax)
            elif isinstance(v, Values):
                param_ranges_tuples[k] = (0, len(v))

        # STEP 1 - Convert param_ranges to SALib Format
        param_ranges_salib = param_tuples_to_salib(param_ranges_tuples)

        # STEP 2 - Create SALib Sample
        salib_sample = saltelli.sample(param_ranges_salib, n, calc_second_order)

        # STEP 3 - Convert back to Agentpy Parameter Dict List and adjust values
        ap_sample = []

        for param_instance in salib_sample:

            parameters = {}
            parameters.update(params)

            for i, key in enumerate(param_ranges_tuples.keys()):
                p = param_instance[i]

                # Convert to integer
                if isinstance(params[key], Range) and params[key].ints:
                    p = int(p) - 1 if p == params[key].vmax+1 else int(p)

                # Convert to value
                if isinstance(params[key], Values):
                    p = int(p) - 1 if p == len(params[key]) else int(p)
                    p = params[key].values[p]  # Find value
                parameters[key] = p

            ap_sample.append(parameters)

        # STEP 4 - Log
        self._log['salib_problem'] = param_ranges_salib
        self._log['calc_second_order'] = calc_second_order

        return ap_sample


