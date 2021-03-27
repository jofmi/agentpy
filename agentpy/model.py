"""
Agentpy Model Module
Content: Main class for agent-based models
"""

import numpy as np
import pandas as pd
from datetime import datetime

from .output import DataDict
from .objects import ApEnv, Agent, Environment
from .network import Network
from .grid import Grid
from .space import Space
from .tools import AttrDict, AgentpyError, make_list
from .lists import ObjList, EnvList


class Model(ApEnv):
    """
    An agent-based model that can hold environments and agents.

    This class can be used as a parent class for custom models.
    Class attributes can be accessed like dictionary items.
    To define the simulation procedure, you can override the methods
    :func:`Model.setup`, :func:`Model.step`,
    :func:`Model.update`, and :func:`Model.end`.
    The perform the simulation, use :func:`Model.run`.

    Attributes:
        name (str): The models' name.
        envs (EnvList): The models' environments.
        agents (AgentList): The models' agents.
        objects (ObjList): The models' agents and environments.
        random (numpy.random.Generator): The models random number generator.
        p (AttrDict): The models' parameters.
        t (int): Current time-step of the model.
        log (dict): The models' recorded variables.
        measures (dict): The models' recorded measures.
        var_keys (list): Names of the model's variables.
        output (DataDict):
            Output data that is generated at the end of a simulation.

    Arguments:
        parameters (dict, optional): Dictionary of model parameters.
            Recommended types for parameters are int, float, str, list,
            numpy.integer, numpy.floating, and numpy.ndarray.
        run_id (int, optional): Number of current run (default None).
        scenario (str, optional): Current scenario (default None).
        **kwargs: Will be forwarded to :func:`Model.setup`.
    """

    def __init__(self, parameters=None, run_id=None, scenario=None, **kwargs):

        self._id_counter = -1
        self._obj_dict = {}  # Objects mapped by their id
        super().__init__(self)  # Model will assign itself id 0

        self.t = 0
        self.run_id = run_id
        self.scenario = scenario

        # Recording
        self._measure_log = {}
        self.output = DataDict()
        self.output.log = {'model_type': self.type,
                           'time_stamp': str(datetime.now())}

        # Private variables
        self._envs = EnvList()
        self._random = np.random.default_rng()
        self._steps = None
        self._parameters = AttrDict(parameters)
        self._stop = False
        self._set_var_ignore()
        self._setup_kwargs = kwargs

    def __repr__(self):
        rep = f"Agent-based model {{"
        keys = ['type', 'agents', 'envs', 'p']
        items = [(k, self[k]) for k in keys]
        items += list(self.__dict__.items())
        for k, v in items:
            if k[0] != '_':
                v = v._short_repr() if '_short_repr' in dir(v) else v
                rep += f"\n'{k}': {v}"
        return rep + '\n}'

    # Properties ------------------------------------------------------------ #

    @property
    def objects(self):
        return ObjList(self.agents + self.envs)

    @property
    def env(self):
        if len(self._envs) == 1:
            return self._envs[0]
        elif len(self._envs) == 0:
            raise AgentpyError(f"{self} has no environment.")
        else:
            raise AgentpyError(f"{self} has more than one environment. Please "
                               "use `Agent.envs` instead of `Agent.env`.")

    @property
    def envs(self):
        return self._envs

    @property
    def random(self):
        return self._random

    @property
    def measures(self):
        return self._measure_log

    # Handling object ids --------------------------------------------------- #

    def get_obj(self, obj_id):
        """ Returns model object with passed object id (int). """
        try:
            return self._obj_dict[obj_id]
        except KeyError:
            raise ValueError(f"Model has no object with obj_id '{obj_id}'.")

    def _new_id(self):
        """ Returns a new unique object id (int). """
        self._id_counter += 1
        return self._id_counter

    # Adding and removing objects ------------------------------------------- #

    def remove_agents(self, agents):
        """ Removes agents from the model, including all environments.
        If used during a loop over an :class:`AgentList`,
        consider using `AgentList.call` with the argument `check_alive=True`
        to avoid calling agents after they have been deleted. """
        for agent in list(make_list(agents)):  # Soft copy as list is changed
            if agent.alive:
                self._agents.remove(agent)
                for env in agent.envs:
                    env._agents.remove(agent)
                agent._envs = EnvList()
                agent._alive = False

    def add_env(self, env_class=Environment, **kwargs):
        """ Adds a new environment to the model.

        Arguments:
            env_class (type, optional):
                The environment class that should be used.
                If none is passed, :class:`Environment` is used.
            **kwargs: Forwarded to the new environment.

        Returns:
              Environment: The new environment.
        """
        new_env = env_class(self.model, **kwargs)
        self.envs.append(new_env)
        return new_env

    def add_network(self, graph=None, agents=None, **kwargs):
        """ Adds a new :class:`Network` environment to the model.
        Arguments are forwarded to the new environment. """
        new_env = Network(self.model, graph=graph, agents=agents, **kwargs)
        self.envs.append(new_env)
        return new_env

    def add_grid(self, shape, **kwargs):
        """ Adds a new :class:`Grid` environment to the model.
        Arguments are forwarded to the new environment. """
        new_env = Grid(self.model, shape=shape, **kwargs)
        self.envs.append(new_env)
        return new_env

    def add_space(self, shape, **kwargs):
        """ Adds a new :class:`Space` environment to the model.
        Arguments are forwarded to the new environment. """
        new_env = Space(self.model, shape=shape, **kwargs)
        self.envs.append(new_env)
        return new_env

    # Recording ------------------------------------------------------------- #

    def measure(self, name, value):
        """ Store a new evaluation measure.

        Evaluation measures are meant to be 'summary statistics' or 'reporters'
        of the whole simulation, and only one value can be stored per run.
        In comparison, variables that are recorded with :func:`Model.record`
        can be recorded multiple times for each time-step and object.

        Arguments:
            name (str): Name of the measure.
            value (int or float): Measured value.

        Examples:

            Store a measure `x` with a value `42`::

                model.measure('x', 42)

            Define a custom model that stores an evaluation measure `sum_id`
            with the sum of all agent ids at the end of the simulation::

                class MyModel(ap.Model):
                    def setup(self):
                        agents = self.add_agents(self.p.agents)
                    def end(self):
                        self.measure('sum_id', sum(self.agents.id))

            Running an experiment over different numbers of agents for this
            model yields the following datadict of measures::

                >>> sample = ap.sample({'agents': (1, 3)}, 3)
                >>> exp = ap.Experiment(MyModel, sample)
                >>> results = exp.run()
                >>> print(results.measures)
                        sum_id
                run_id
                0            1
                1            3
                2            6
        """
        self._measure_log[name] = [value]

    # Placeholder methods for custom simulation methods --------------------- #

    def setup(self, **kwargs):
        """ Defines the model's actions before the first simulation step.
        Can be overwritten and used to initiate agents and environments."""
        pass

    def step(self):
        """ Defines the model's actions during each simulation step.
        Can be overwritten and used to set the models' main dynamics."""
        pass

    def update(self):
        """ Defines the model's actions after setup and each simulation step.
        Can be overwritten and used for the recording of dynamic variables. """
        pass

    def end(self):
        """ Defines the model's actions after the last simulation step.
        Can be overwritten and used for final calculations and measures."""
        pass

    # Simulation routines (in line with ipysimulate) ------------------------ #

    def set_parameters(self, parameters):
        """ Adds or updates passed parameters. """
        self._parameters.update(parameters)

    def run_setup(self, steps=None, seed=None):
        """ Sets up time-step 0 of the simulation.
        Prepares steps and a random number generator,
        and then calls :func:`Model.setup` and :func:`Model.update`. """

        # Prepare random generator
        if not seed and 'seed' in self.p:
            seed = self.p['seed']  # Take seed from parameters
        if seed:
            self._random = np.random.default_rng(seed=seed)

        # Prepare steps
        if steps is None:
            self._steps = self.p['steps'] if 'steps' in self.p else 1000
        else:
            self._steps = steps

        # Initiate simulation
        self._stop = False

        # Execute setup and first update
        self.setup(**self._setup_kwargs)
        self.update()

        # Stop simulation if t too high
        if self.t >= self._steps:
            self._stop = True

    def run_step(self):
        """ Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`."""
        self.t += 1
        self.step()
        self.update()
        if self.t >= self._steps:
            self._stop = True

    def stop(self):
        """ Stops :meth:`Model.run` during an active simulation. """
        self._stop = True

    @property
    def is_running(self):
        """ Indicates whether the model is currently running (bool). """
        return not self._stop

    def reset(self):
        """ Reset model to initial conditions and call setup. """
        self.__init__(parameters=self.p,
                      run_id=self.run_id,
                      scenario=self.scenario,
                      **self._setup_kwargs)

    # Data management ------------------------------------------------------- #

    def create_output(self):
        """ Generates a :class:`DataDict` with dataframes of all recorded
        variables and measures, which will be stored in :obj:`Model.output`.
        """

        def output_from_obj_list(self, obj_list, columns):
            # Aggregate logs per object type
            obj_types = {}
            for obj in obj_list:

                if obj.log:  # Check for variables

                    # Add object id/key to object log
                    obj.log['obj_id'] = [obj.id] * len(obj.log['t'])

                    # Initiate object type if new
                    obj_type = type(obj).__name__

                    if obj_type not in obj_types.keys():
                        obj_types[obj_type] = {}

                    # Add object log to aggr. log
                    for k, v in obj.log.items():
                        if k not in obj_types[obj_type]:
                            obj_types[obj.type][k] = []
                        obj_types[obj_type][k].extend(v)

            # Transform logs into dataframes
            for obj_type, log in obj_types.items():
                df = pd.DataFrame(log)
                for k, v in columns.items():
                    df[k] = v  # Set additional index columns
                df = df.set_index(list(columns.keys()) + ['obj_id', 't'])
                self.output['variables'][obj_type] = df

        # 0 - Document parameters
        if self.p:
            self.output['parameters'] = self.p

        # 1 - Define additional index columns
        columns = {}
        if self.run_id is not None:
            columns['run_id'] = self.run_id
        if self.scenario is not None:
            columns['scenario'] = self.scenario

        # 2 - Create measure output
        if self._measure_log:
            d = self._measure_log
            for key, value in columns.items():
                d[key] = value
            df = pd.DataFrame(d)
            if columns:
                df = df.set_index(list(columns.keys()))
            self.output['measures'] = df

        # 3 - Create variable output
        self.output['variables'] = DataDict()

        # 3.1 - Create variable output for objects
        output_from_obj_list(self, self.agents, columns)
        output_from_obj_list(self, self.envs, columns)

        # 3.2 - Create variable output for model
        if self.log:
            df = pd.DataFrame(self.log)
            # df['obj_id'] = 'model'
            for k, v in columns.items():
                df[k] = v
            df = df.set_index(list(columns.keys()) + ['t'])  # 'obj_id',

            if self.output['variables']:
                self.output['variables'][self.type] = df
            else:
                self.output['variables'] = df  # No subdict if only model vars

        # 3.3 - Remove variable dict if empty (i.e. nothing has been added)
        elif not self.output['variables']:
            del self.output['variables']

    # Main simulation method for direct use --------------------------------- #

    def run(self, steps=None, seed=None, display=True):
        """ Executes the simulation of the model.

        It starts by calling :func:`Model.run_setup` and then calls
        :func:`Model.run_step` until the method :func:`Model.stop` is called
        or `steps` is reached. After that, :func:`Model.end` and
        :func:`Model.create_output` are called.

        Arguments:
            steps (int, optional):
                Maximum number of steps for the simulation to run.
                If none is given, the parameter 'Model.p.steps' will be used.
                If there is no such parameter, 'steps' will be set to 1000.
            seed (int, optional):
                Seed to set for :obj:`Model.random`
                at the beginning of the simulation.
                If none is given, the parameter 'Model.p.seed' will be used.
                If there is no such parameter, as random seed will be set.
            display (bool, optional):
                Whether to display simulation progress (default True).

        Returns:
            DataDict: Recorded model data,
            which can also be found in :attr:`Model.output`.
        """

        dt0 = datetime.now()  # Time-Stamp
        self.run_setup(steps, seed)
        while not self._stop:
            self.run_step()
            if display:
                print(f"\rCompleted: {self.t} steps", end='')
        self.end()
        self.create_output()
        self.output.log['run_time'] = ct = str(datetime.now() - dt0)
        self.output.log['steps'] = self.t

        if display:
            print(f"\nRun time: {ct}\nSimulation finished")

        return self.output

