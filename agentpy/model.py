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
from .lists import ObjList


class Model(ApEnv):
    """
    An agent-based model that can hold environments and agents.

    This class can be used as a parent class for custom models.
    Class attributes can be accessed like dictionary items.
    To define the procedures of a simulation, override the methods
    :func:`Model.setup`, :func:`Model.step`,
    :func:`Model.update`, and :func:`Model.end`.
    See :func:`Model.run` for more information on the simulation procedure.

    Attributes:
        name (str): The models' name.
        envs (EnvList): The models' environments.
        agents (AgentList): The models' agents.
        p (AttrDict): The models' parameters.
        t (int): Current time-step of the model.
        log (dict): The models' recorded variables.
        output (DataDict): Output data after simulation.

    Arguments:
        parameters (dict, optional): Dictionary of model parameters.
            Recommended types for parameters are int, float, str, list,
            numpy.integer, numpy.floating, and numpy.ndarray.
            Other types might cause errors.
        run_id (int, optional): Number of current run (default None).
        scenario (str, optional): Current scenario (default None).
        **kwargs: Will be forwarded to :func:`Model.setup`
    """

    def __init__(self, parameters=None, run_id=None, scenario=None, **kwargs):

        self._id_counter = -1
        self._obj_dict = {}  # Objects mapped by their id
        super().__init__(self)  # Model will be have id 0

        self.t = 0
        self.run_id = run_id
        self.scenario = scenario

        # Recording
        self._measure_log = {}
        self.output = DataDict()
        self.output.log = {'model_type': self.type,
                           'time_stamp': str(datetime.now())}

        # Private variables
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

    @property
    def objects(self):
        """ Returns a list of all model objects (agents and environments). """
        return ObjList(self.agents + self.envs)

    @property
    def random(self):
        """ Returns the models random number generator
        of type :class:`numpy.random.Generator`. """
        return self._random

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

    def add_env(self, env_class=Environment, **kwargs):
        """ Creates a new environment. """
        new_env = env_class(self.model, **kwargs)
        self.envs.append(new_env)
        return new_env

    def add_network(self, graph=None, agents=None, **kwargs):
        """ Creates a new environment with a network.
        Arguments are forwarded to :class:`Network`. """
        new_env = Network(self.model, graph=graph, agents=agents, **kwargs)
        self.envs.append(new_env)
        return new_env

    def add_grid(self, shape, **kwargs):
        """ Creates a new environment with a spatial grid.
        Arguments are forwarded to :class:`Grid`. """
        new_env = Grid(self.model, shape=shape, **kwargs)
        self.envs.append(new_env)
        return new_env

    def add_space(self, shape, **kwargs):
        """ Creates a new environment with a continuous space.
        Arguments are forwarded to :class:`Space`. """
        new_env = Space(self.model, shape=shape, **kwargs)
        self.envs.append(new_env)
        return new_env

    def measure(self, measure, value):
        """ Records an evaluation measure. """
        self._measure_log[measure] = [value]

    # Main simulation functions

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

    def stop(self):
        """ Stops :meth:`Model.run` during an active simulation. """
        self._stop = True

    def _setup_run(self, steps=None, seed=None):
        """ Prepare round 0 of a simulation.
        See Model.run() for more informatin. """

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

    def _make_step(self):
        """ Proceed simulation by one step. """
        self.t += 1
        self.step()
        self.update()
        if self.t >= self._steps:
            self._stop = True

    def run(self, steps=None, seed=None, display=True):
        """ Executes the simulation of the model.

        The simulation proceeds as follows.
        It starts by calling :func:`Model.setup` and :func:`Model.update`.
        After that, :attr:`Model.t` is increased by 1 and
        :func:`Model.step` and :func:`Model.update` are called.
        This step is repeated until the method :func:`Model.stop` is called
        or steps is reached. After the last step, :func:`Model.end` is called.

        Arguments:
            steps (int, optional):
                Maximum number of steps for the simulation to run.
                If none is given, the parameter 'Model.p.steps' will be used.
                If there is no such parameter, 'steps' will be set to 1000.
            seed (int, optional):
                Seed to set for :obj:`Model.random` at the beginning of the simulation.
                If none is given, the parameter 'Model.p.seed' will be used.
                If there is no such parameter, as random seed will be set.
            display (bool, optional):
                Whether to display simulation progress (default True).

        Returns:
            DataDict: Recorded model data,
            which can also be found in :attr:`Model.output`.
        """

        dt0 = datetime.now()  # Time-Stamp
        self._setup_run(steps, seed)

        while not self._stop:
            self._make_step()
            if display:
                print(f"\rCompleted: {self.t} steps", end='')

        self.end()
        self._create_output()
        self.output.log['run_time'] = ct = str(datetime.now() - dt0)
        self.output.log['steps'] = self.t

        if display:
            print(f"\nRun time: {ct}\nSimulation finished")

        return self.output

    def _create_output(self):
        """ Generates an 'output' dictionary out of object logs. """

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
