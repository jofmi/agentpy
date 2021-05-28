"""
Agentpy Experiment Module
Content: Experiment class
"""

import pandas as pd
import random as rd

from os import sys

from .version import __version__
from datetime import datetime, timedelta
from .tools import make_list
from .datadict import DataDict
from .sample import Sample, Range, IntRange, Values

class Experiment:
    """ Experiment that can run an agent-based model
    over for multiple iterations and parameter combinations
    and generate combined output data.

    Arguments:
        model (type):
            The model class for the experiment to use.
        sample (dict or list of dict or Sample, optional):
            Parameter combination(s) to test in the experiment (default None).
        iterations (int, optional):
            How often to repeat every parameter combination (default 1).
        record (bool, optional):
            Keep the record of dynamic variables (default False).
        randomize (bool, optional):
            Generate different random seeds for every iteration (default True).
            If True, the parameter 'seed' will be used to initialize a random
            seed generator for every parameter combination in the sample.
            If False, the same seed will be used for every iteration.
            If no parameter 'seed' is defined, this option has no effect.
            For more information, see :doc:`guide_random` .
        **kwargs:
            Will be forwarded to all model instances created by the experiment.

    Attributes:
        output(DataDict): Recorded experiment data
    """

    def __init__(self, model_class, sample=None, iterations=1,
                 record=False, randomize=True, **kwargs):

        self.model = model_class
        self.output = DataDict()
        self.iterations = iterations
        self.record = record
        self._model_kwargs = kwargs
        self.name = model_class.__name__

        # Prepare sample
        if isinstance(sample, Sample):
            self.sample = list(sample)
            self._sample_log = sample._log
        else:
            self.sample = make_list(sample, keep_none=True)
            self._sample_log = None

        # Prepare runs
        len_sample = len(self.sample)
        iter_range = range(iterations) if iterations > 1 else [None]
        sample_range = range(len_sample) if len_sample > 1 else [None]
        self.run_ids = [(sample_id, iteration)
                        for sample_id in sample_range
                        for iteration in iter_range]
        self.n_runs = len(self.run_ids)

        # Prepare seeds
        if randomize and sample is not None \
                and any(['seed' in p for p in self.sample]):
            if len_sample > 1:
                rngs = [rd.Random(p['seed'])
                        if 'seed' in p else rd.Random() for p in self.sample]
                self._random = {
                    (sample_id, iteration): rngs[sample_id].getrandbits(128)
                    for sample_id in sample_range
                    for iteration in iter_range
                }
            else:
                p = list(self.sample)[0]
                seed = p['seed']
                ranges = (Range, IntRange, Values)
                if isinstance(seed, ranges):
                    seed = seed.vdef
                rng = rd.Random(seed)
                self._random = {
                    (None, iteration): rng.getrandbits(128)
                    for iteration in iter_range
                }
        else:
            self._random = None

        # Prepare output
        self.output.info = {
            'model_type': model_class.__name__,
            'time_stamp': str(datetime.now()),
            'agentpy_version': __version__,
            'python_version': sys.version[:5],
            'experiment': True,
            'scheduled_runs': self.n_runs,
            'completed': False,
            'random': randomize,
            'record': record,
            'sample_size': len(self.sample),
            'iterations': iterations
        }
        self._parameters_to_output()

    def _parameters_to_output(self):
        """ Document parameters (separately for fixed & variable). """
        df = pd.DataFrame(self.sample)
        df.index.rename('sample_id', inplace=True)
        fixed_pars = {}
        for col in df.columns:
            s = df[col]
            if len(s.unique()) == 1:
                fixed_pars[s.name] = df[col][0]
                df.drop(col, inplace=True, axis=1)
        self.output['parameters'] = DataDict()
        if fixed_pars:
            self.output['parameters']['constants'] = fixed_pars
        if not df.empty:
            self.output['parameters']['sample'] = df
        if self._sample_log:
            self.output['parameters']['log'] = self._sample_log

    @staticmethod
    def _add_single_output_to_combined(single_output, combined_output):
        """Append results from single run to combined output.
        Each key in single_output becomes a key in combined_output.
        DataDicts entries become dicts with lists of values.
        Other entries become lists of values. """
        for key, value in single_output.items():
            if key in ['parameters', 'info']:  # Skip parameters & info
                continue
            if isinstance(value, DataDict):  # Handle subdicts
                if key not in combined_output:  # New key
                    combined_output[key] = {}  # as dict
                for obj_type, obj_df in single_output[key].items():
                    if obj_type not in combined_output[key]:  # New subkey
                        combined_output[key][obj_type] = []  # as list
                    combined_output[key][obj_type].append(obj_df)
            else:  # Handle other output types
                if key not in combined_output:  # New key
                    combined_output[key] = []  # as list
                combined_output[key].append(value)

    def _combine_dataframes(self, combined_output):
        """ Combines data from combined output.
        Dataframes are combined with concat.
        Dicts are transformed to DataDict.
        Other objects are kept as original.
        Combined data is written to self.output. """
        for key, values in combined_output.items():
            if values and all([isinstance(value, pd.DataFrame)
                               for value in values]):
                self.output[key] = pd.concat(values)  # Df are combined
            elif isinstance(values, dict):  # Dict is transformed to DataDict
                self.output[key] = DataDict()
                for sk, sv in values.items():
                    if all([isinstance(v, pd.DataFrame) for v in sv]):
                        self.output[key][sk] = pd.concat(sv)  # Df are combined
                    else:  # Other objects are kept as original TODO TESTS
                        self.output[key][sk] = sv
            elif key != 'info':  # Other objects are kept as original TODO TESTS
                self.output[key] = values

    def _single_sim(self, run_id):
        """ Perform a single simulation."""
        sample_id = 0 if run_id[0] is None else run_id[0]
        parameters = self.sample[sample_id]
        model = self.model(parameters, _run_id=run_id, **self._model_kwargs)
        if self._random:
            results = model.run(display=False, seed=self._random[run_id])
        else:
            results = model.run(display=False)
        if 'variables' in results and self.record is False:
            del results['variables']  # Remove dynamic variables from record
        return results

    def run(self, pool=None, display=True):
        """ Perform the experiment.
        The simulation will run the model once for each set of parameters
        and will repeat this process for the set number of iterations.
        Simulation results will be stored in `Experiment.output`.

        Arguments:
            pool (multiprocessing.Pool, optional):
                Pool of active processes for parallel processing.
                If none is passed, normal processing is used.
            display (bool, optional):
                Display simulation progress (default True).

        Returns:
            DataDict: Recorded experiment data.

        Examples:

            To run a normal experiment::

                exp = ap.Experiment(MyModel, parameters)
                results = exp.run()

            To use parallel processing::

                import multiprocessing as mp
                if __name__ ==  '__main__':
                    exp = ap.Experiment(MyModel, parameters)
                    pool = mp.Pool(mp.cpu_count())
                    results = exp.run(pool)
        """

        if display:
            n_runs = self.n_runs
            print(f"Scheduled runs: {n_runs}")
        t0 = datetime.now()  # Time-Stamp Start
        combined_output = {}

        # Normal processing
        if pool is None:
            i = -1
            for run_id in self.run_ids:
                self._add_single_output_to_combined(
                    self._single_sim(run_id), combined_output)
                if display:
                    i += 1
                    td = (datetime.now() - t0).total_seconds()
                    te = timedelta(seconds=int(td / (i + 1)
                                               * (n_runs - i - 1)))
                    print(f"\rCompleted: {i + 1}, "
                          f"estimated time remaining: {te}", end='')
            if display:
                print("")  # Because the last print ended without a line-break

        # Parallel processing
        else:
            if display:
                print(f"Using parallel processing.")
                print(f"Active processes: {pool._processes}")
            output_list = pool.map(self._single_sim, self.run_ids)
            for single_output in output_list:
                self._add_single_output_to_combined(
                    single_output, combined_output)

        self._combine_dataframes(combined_output)
        self.end()
        self.output.info['completed'] = True
        self.output.info['run_time'] = ct = str(datetime.now() - t0)

        if display:
            print(f"Experiment finished\nRun time: {ct}")

        return self.output

    def end(self):
        """ Defines the experiment's actions after the last simulation.
        Can be overwritten for final calculations and reporting."""
        pass