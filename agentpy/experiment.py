"""
Agentpy Experiment Module
Content: Experiment class
"""

import pandas as pd

from datetime import datetime, timedelta
from .tools import make_list
from .output import DataDict


class Experiment:
    """ Experiment for an agent-based model.
    Allows for multiple iterations, parameter samples, and distict scenarios.

    Arguments:
        model(type): The model class type that the experiment should use.
        parameters(dict or list of dict, optional): Parameter dictionary
            or sample (list of parameter dictionaries) (default None).
        name(str, optional): Name of the experiment (default model.name).
        scenarios(str or list, optional): Experiment scenarios (default None).
        iterations(int, optional): Experiment repetitions (default 1).
        record(bool, optional): Record dynamic variables (default False).

    Attributes:
        output(DataDict): Recorded experiment data
    """  # TODO Repeat arguments in attribute list? / Type hint for model?

    def __init__(self, model, parameters=None, name=None, scenarios=None,
                 iterations=1, record=False):

        # Experiment objects
        self.model = model
        self.parameters = parameters
        self.output = DataDict()

        # Experiment settings
        if name:
            self.name = name
        else:
            self.name = model.__name__
        self.scenarios = scenarios
        self.iterations = iterations
        self.record = record

        # Log
        self.output.log = {'name': self.name,
                           'time_stamp': str(datetime.now()),
                           'iterations': iterations}
        if scenarios:
            self.output.log['scenarios'] = scenarios

    def run(self, display=True):
        """ Executes the simulation of the experiment.

        The simulation will run the model once for each set of parameters
        and will repeat this process for the set number of iterations.
        Simulation results are stored in ``Experiment.output``.

        Arguments:
            display(bool,optional): Display simulation progress (default True).

        Returns:
            DataDict: Recorded experiment data.
        """

        # Transform input into iterable lists if only a single value is given
        # keep_none assures that make_list(None) returns iterable [None]
        parameter_sample = make_list(self.parameters, keep_none=True)
        scenarios = make_list(self.scenarios, keep_none=True)

        # Prepare runs
        runs = parameter_sample * self.iterations
        n_runs = self.output.log['n_runs'] = len(runs)

        # Document parameters (seperately for fixed & variable)
        df = pd.DataFrame(parameter_sample)
        df.index.rename('sample_id', inplace=True)
        fixed_pars = {}
        for col in df.columns:
            s = df[col]
            # TODO Error if parameters are unhashable (e.g. list,dict)
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

        # Perform experiment

        if display:
            print(f"Scheduled runs: {n_runs}")
        t0 = datetime.now()  # Time-Stamp Start

        combined_output = {}

        for i, parameters in enumerate(runs):

            for scenario in scenarios:

                # Run model for current parameters & scenario
                single_output = self.model(
                    parameters, run_id=i, scenario=scenario).run(display=False)

                # Append results to experiment output
                for key, value in single_output.items():

                    # Skip parameters & log
                    if key in ['parameters', 'log']:
                        continue

                    # Skip variables if record is False
                    if key == 'variables' and not self.record:
                        continue

                    # Handle variable subdicts
                    if key == 'variables' and isinstance(value, DataDict):

                        if key not in combined_output:
                            combined_output[key] = {}

                        for obj_type, obj_df in single_output[key].items():

                            if obj_type not in combined_output[key]:
                                combined_output[key][obj_type] = []

                            combined_output[key][obj_type].append(obj_df)

                    # Handle other output types
                    else:
                        if key not in combined_output:
                            combined_output[key] = []
                        combined_output[key].append(value)

            if display:
                td = (datetime.now() - t0).total_seconds()
                te = timedelta(seconds=int(td / (i + 1) * (n_runs - i - 1)))
                print(f"\rCompleted: {i + 1}, estimated time remaining: {te}",
                      end='')

        # Combine dataframes
        for key, values in combined_output.items():
            if values and all([isinstance(value, pd.DataFrame)
                               for value in values]):
                self.output[key] = pd.concat(values)
            elif isinstance(values, dict):  # Create SubDataDict
                self.output[key] = DataDict()
                for sk, sv in values.items():
                    self.output[key][sk] = pd.concat(sv)
            elif key != 'log':
                self.output[key] = values

        self.output.log['run_time'] = ct = str(datetime.now() - t0)

        if display:
            print(f"\nRun time: {ct}\nSimulation finished")

        return self.output
