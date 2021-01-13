"""
Agentpy Experiment Module
Content: Experiment class
"""

import pandas as pd
import ipywidgets
import IPython

from datetime import datetime, timedelta
from .tools import make_list
from .output import DataDict


class Experiment:
    """ Experiment for an agent-based model.
    Allows for multiple iterations, parameter samples, scenario comparison,
    and parallel processing. See :func:`Experiment.run` for standard
    simulations and :func:`Experiment.interactive` for interactive output.

    Arguments:
        model_class(type): The model class type that the experiment should use.
        parameters(dict or list of dict, optional):
            Parameter dictionary or sample (default None).
        name(str, optional): Name of the experiment (default model.name).
        scenarios(str or list, optional): Experiment scenarios (default None).
        iterations(int, optional): Experiment repetitions (default 1).
        record(bool, optional):
            Whether to keep the record of dynamic variables (default False).
            Note that this does not affect evaluation measures.
        **kwargs: Will be forwarded to the creation of every model instance
            during the experiment.

    Attributes:
        output(DataDict): Recorded experiment data
    """

    def __init__(self, model_class, parameters=None, name=None, scenarios=None,
                 iterations=1, record=False, **kwargs):

        self.model = model_class
        self.output = DataDict()
        self.iterations = iterations
        self.record = record
        self._model_kwargs = kwargs

        if name:
            self.name = name
        else:
            self.name = model_class.__name__

        # Transform input into iterable lists if only a single value is given
        # keep_none assures that make_list(None) returns iterable [None]
        self.scenarios = make_list(scenarios, keep_none=True)
        self.parameters = make_list(parameters, keep_none=True)
        self._parameters_to_output()  # Record parameters

        # Log
        self.output.log = {'name': self.name,
                           'model_type': model_class.__name__,
                           'time_stamp': str(datetime.now()),
                           'iterations': iterations}
        if scenarios:
            self.output.log['scenarios'] = scenarios

        # Prepare runs
        self.parameters_per_run = self.parameters * self.iterations
        self.number_of_runs = len(self.parameters_per_run)

    def _parameters_to_output(self):
        """ Document parameters (seperately for fixed & variable). """
        df = pd.DataFrame(self.parameters)
        df.index.rename('sample_id', inplace=True)
        fixed_pars = {}
        for col in df.columns:
            s = df[col]
            if len(s.unique()) == 1:
                fixed_pars[s.name] = df[col][0]
                df.drop(col, inplace=True, axis=1)
        if fixed_pars and df.empty:
            self.output['parameters'] = fixed_pars
        elif not fixed_pars and not df.empty:
            self.output['parameters'] = df
        else:
            self.output['parameters'] = DataDict({
                'fixed': fixed_pars,
                'varied': df
            })

    def _add_single_output_to_combined(self, single_output, combined_output):
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

    def _single_sim(self, sim_id):
        """ Perform a single simulation."""
        sc_id = sim_id % len(self.scenarios)
        run_id = (sim_id - sc_id) // len(self.scenarios)
        model = self.model(
            self.parameters_per_run[run_id],
            run_id=run_id,
            scenario=self.scenarios[sc_id],
            **self._model_kwargs)
        results = model.run(display=False)
        if 'variables' in results and self.record is False:
            del results['variables']  # Remove dynamic variables from record
        return results

    def run(self, pool=None, display=True):
        """ Executes a multi-run experiment.

        The simulation will run the model once for each set of parameters
        and will repeat this process for the set number of iterations.
        Parallel processing is possible if a `pool` is passed.
        Simulation results will be stored in `Experiment.output`.

        Arguments:
            pool(multiprocessing.Pool, optional):
                Pool of active processes for parallel processing.
                If none is passed, normal processing is used.
            display(bool, optional):
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

        sim_ids = list(range(self.number_of_runs * len(self.scenarios)))
        n_sims = len(sim_ids)
        if display:
            print(f"Scheduled runs: {n_sims}")
        t0 = datetime.now()  # Time-Stamp Start
        combined_output = {}

        if pool is None:  # Normal processing
            for sim_id in sim_ids:
                self._add_single_output_to_combined(
                    self._single_sim(sim_id), combined_output)
                if display:
                    td = (datetime.now() - t0).total_seconds()
                    te = timedelta(seconds=int(td / (sim_id + 1)
                                               * (n_sims - sim_id - 1)))
                    print(f"\rCompleted: {sim_id + 1}, "
                          f"estimated time remaining: {te}", end='')
            if display:
                print("")  # Because the last print ended without a line-break
        else:  # Parallel processing
            if display:
                print(f"Active processes: {pool._processes}")
            output_list = pool.map(self._single_sim, sim_ids)
            for single_output in output_list:
                self._add_single_output_to_combined(
                    single_output, combined_output)

        self._combine_dataframes(combined_output)
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
            parameters = dict(self.parameters[0])
            parameters.update(param_updates)
            temp_model = self.model(parameters, **self._model_kwargs)
            temp_model.run()
            IPython.display.clear_output()
            plot(temp_model, *args, **kwargs)

        # Get variable parameters
        var_pars = self.output._combine_pars(varied=True, fixed=False)

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
