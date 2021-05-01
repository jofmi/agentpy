"""
Agentpy Experiment Module
Content: Experiment class
"""

import pandas as pd
import ipywidgets
import IPython
import random as rd

from os import sys

from .version import __version__
from datetime import datetime, timedelta
from .tools import make_list
from .datadict import DataDict
from .sample import Sample


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
        random (bool, optional):
            Choose random seeds for every new iteration (default False).
            The seed for the random number generator will be taken from the
            experiments's current parameter combination.
            Note that if there is no parameter 'seed',
            iterations will have random seeds even if this is False.
        **kwargs:
            Will be forwarded to all model instances created by the experiment.

    Attributes:
        output(DataDict): Recorded experiment data
    """

    def __init__(self, model_class, sample=None, iterations=1,
                 record=False, random=False, **kwargs):

        self.model = model_class
        self.output = DataDict()
        self.iterations = iterations
        self.record = record
        self._model_kwargs = kwargs
        self.name = model_class.__name__

        # Prepare sample
        if isinstance(sample, Sample):
            self.sample = list(sample)
            self.sample_log = sample._log
        else:
            self.sample = make_list(sample, keep_none=True)
            self.sample_log = None

        # Prepare runs
        combos = len(self.sample)
        iter_range = range(iterations) if iterations > 1 else [None]
        sample_range =range(combos) if combos > 1 else [None]
        self.run_ids = [(sample_id, iteration)
                        for sample_id in sample_range
                        for iteration in iter_range]
        self.n_runs = len(self.run_ids)

        # Prepare seeds
        if random:
            rngs = [rd.Random(p['seed'])
                    if 'seed' in p else rd.Random()
                    for p in sample]
            self._random = {
                (sample_id, iteration): rngs[sample_id].getrandbits(128)
                for sample_id in range(len(self.sample))
                for iteration in range(iterations)
            }
        else:
            self._random = None

        # Prepare output
        self.output.log = {
            'model_type': model_class.__name__,
            'time_stamp': str(datetime.now()),
            'agentpy_version': __version__,
            'python_version': sys.version[:5],
            'multi_run': True,
            'scheduled_runs': self.n_runs,
            'completed': False,
            'random': random,
            'record': record,
            'sample_size': len(self.sample),
            'iterations': iterations
        }
        self._parameters_to_output()

    def _parameters_to_output(self):
        """ Document parameters (seperately for fixed & variable). """
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
        if self.sample_log:
            self.output['parameters']['sample_log'] = self.sample_log

    @staticmethod
    def _add_single_output_to_combined(single_output, combined_output):
        """Append results from single run to combined output.
        Each key in single_output becomes a key in combined_output.
        DataDicts entries become dicts with lists of values.
        Other entries become lists of values. """
        for key, value in single_output.items():
            if key in ['parameters', 'log']:  # Skip parameters & log
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
            elif key != 'log':  # Other objects are kept as original TODO TESTS
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
        self.output.log['completed'] = True
        self.output.log['run_time'] = ct = str(datetime.now() - t0)

        if display:
            print(f"Experiment finished\nRun time: {ct}")

        return self.output

    def interactive(self, plot, *args, **kwargs):
        """
        Displays interactive output for Jupyter notebooks,
        using :mod:`IPython` and :mod:`ipywidgets`.
        A slider will be shown for varied parameters.
        Every time a parameter value is changed on the slider,
        the experiment will re-run the model and pass it
        to the 'plot' function.

        Arguments:
            plot: Function that takes a model instance as input
                and prints or plots the desired output..
            *args: Will be forwarded to 'plot'.
            **kwargs: Will be forwarded to 'plot'.

        Returns:
            ipywidgets.HBox: Interactive output widget
            
        Examples:
            The following example uses a custom model :class:`MyModel`
            and creates a slider for the parameters 'x' and 'y',
            both of which can be varied interactively over 10 different values.
            Every time a value is changed, the experiment will simulate the
            model with the new parameters and pass it to the plot function::
            
                def plot(model):
                    # Display interactive output here
                    print(model.output)
                    
                param_ranges = {'x': (0, 10), 'y': (0., 1.)}
                sample = ap.sample(param_ranges, n=10)
                exp = ap.Experiment(MyModel, sample)
                exp.interactive(plot)
        """

        def var_run(**param_updates):
            """ Display plot for updated parameters. """
            IPython.display.clear_output()
            parameters = dict(self.sample[0])
            parameters.update(param_updates)
            temp_model = self.model(parameters, **self._model_kwargs)
            temp_model.run()
            IPython.display.clear_output()
            plot(temp_model, *args, **kwargs)

        print(self.output.parameters)

        # Get variable parameters
        var_pars = self.output._combine_pars(sample=True, constants=False)

        # Create widget dict
        widget_dict = {}
        for par_key in list(var_pars):
            par_list = list(var_pars[par_key])

            widget_dict[par_key] = ipywidgets.SelectionSlider(
                options=par_list,
                value=par_list[0],
                description=par_key,
                continuous_update=False,
                style=dict(description_width='initial'),
                layout={'width': '300px'}
            )

        widgets_left = ipywidgets.VBox(list(widget_dict.values()))
        output_right = ipywidgets.interactive_output(var_run, widget_dict)

        return ipywidgets.HBox([widgets_left, output_right])
