"""
Agentpy Framework Module
Content: Classes for agent-based models, environemnts, and agents
"""

import pandas as pd
import networkx as nx
import random as rd

from datetime import datetime
from itertools import product

from .output import DataDict
from .tools import AttrDict, AgentpyError, make_list, make_matrix


class ApObj:
    """ Agentpy base-class for objects of agent-based models."""

    def __init__(self, model):
        self._log = {}
        self._model = model
        self._envs = EnvDict()
        self._var_ignore = []

    @property
    def type(self):
        """Class name of the object (str)."""
        return type(self).__name__

    @property
    def var_keys(self):
        """The object's variables (list of str)."""
        return [k for k in self.__dict__.keys()
                if k[0] != '_'
                and k not in self._var_ignore]

    @property
    def p(self):
        return self._model._parameters

    @property
    def log(self):
        return self._log

    @property
    def model(self):
        return self._model

    @property
    def envs(self):
        return self._envs

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def _set_var_ignore(self):
        """Store current attributes to seperate them from custom variables"""
        self._var_ignore = [k for k in self.__dict__.keys() if k[0] != '_']

    def record(self, var_keys, value=None):
        """ Records an objects variables.

        Arguments:
            var_keys (str or list of str):
                Names of the variables to be recorded.
            value (optional): Value to be recorded.
                The same value will be used for all `var_keys`.
                If none is given, the values of object attributes
                with the same name as each var_key will be used.

        Examples:

            Record the existing attributes ``x`` and ``y`` of an object ``a``::

                a.record(['x', 'y'])

            Record a variable ``z`` with the value ``1`` for an object ``a``::

                a.record('z', 1)

            Record all variables of an object::

                a.record(a.var_keys)
        """

        for var_key in make_list(var_keys):

            # Create empty lists
            if 't' not in self.log:
                self.log['t'] = []
            if var_key not in self.log:
                self.log[var_key] = [None] * len(self.log['t'])

            if self.model.t not in self.log['t']:

                # Create empty slot for new documented time step
                for v in self.log.values():
                    v.append(None)

                # Store time step
                self.log['t'][-1] = self.model.t

            if value is None:
                v = getattr(self, var_key)
            else:
                v = value

            self.log[var_key][-1] = v

    def setup(self, **kwargs):
        """This empty method is called automatically at the objects' creation.
        Can be overwritten in custom sub-classes
        to define initial attributes and actions.

        Arguments:
            **kwargs: Keyword arguments that have been passed to
                :class:`Agent` or :func:`Model.add_agents`.
                If the original setup method is used,
                they will be set as attributes of the object.

        Examples:
            The following setup initializes an object with three variables::

                def setup(self, y):
                    self.x = 0  # Value defined locally
                    self.y = y  # Value defined in kwargs
                    self.z = self.p.z  # Value defined in parameters
        """

        for k, v in kwargs.items():
            setattr(self, k, v)


# Level 3 - Agent class

class Agent(ApObj):
    """ Individual agent of an agent-based model.

    This class can be used as a parent class for custom agent types.

    All agentpy model objects call the method :func:`setup()` after creation,
    can access class attributes like dictionary items,
    and can be removed from the model with the `del` statement.

    Attributes:
        model (Model): Model instance.
        p (AttrDict): Model parameters.
        envs (EnvDict): Environments of the agent.
        log (dict): Recorded variables.
        id (int): Unique identifier.

    Arguments:
        model (Model): Instance of the current model.
        envs (dict or EnvDict, optional): The agents' initial environments.

    """

    def __init__(self, model, envs=None, **kwargs):
        super().__init__(model)
        self.id = model._new_id()
        if envs:  # Add environments
            self.envs.update(envs)
        self._set_var_ignore()
        self.setup(**kwargs)

    def __repr__(self):
        s = f"Agent {self.id}"
        t = type(self).__name__
        if t != 'Agent':
            s += f" ({t})"
        return s

    def __getattr__(self, item):
        raise AttributeError(f"{self} has no attribute '{item}'")

    def __del__(self):
        self.model.agents.remove(self)

    def position(self, env_key=None):
        """ Returns the agents' position from a grid.

        Arguments:
            env_key(str, optional): Name of environment of type :class:`grid`.
                If none is given, the first grid in ``Agent.envs`` is selected.
        """

        # Default environment
        if env_key is None:
            grids = [k for k, v in self.envs.items() if v.topology in ['grid']]
            try:
                env_key = grids[0]
            except IndexError:
                raise AgentpyError(
                    f"Agent {self.id} has no environment of type 'grid'")

        try:
            return self.envs[env_key]._positions[self]
        except KeyError:
            raise AgentpyError(
                f"Agent {self.id} has no environment '{env_key}'")

    def move(self, position, env_key=None, relative=True):
        """ Changes the agents' location in the selected environment.

        Arguments:
            position (list of int): Position to move to. If relative is True,
                position is added to the agent's current position.
            env_key (str, optional): Grid environment in which to move.
                If none is given, the agents first grid is selected.
            relative (bool, optional): See description of 'position'.
        """

        if relative:
            old_pos = self.possition(env_key)
            position = [p + c for p, c in zip(old_pos, position)]

        self.envs[env_key].change_pos(self, position)

    def neighbors(self, env_keys=None):
        """ Returns the agents' neighbor's from grids and/or networks.

        Arguments:
            env_keys(str or list of str, optional): Names of environments
                from the agent with type :class:`network` or :class:`grid`.
                If none are given, all such environments are selected.

        Returns:
            AgentList
        """

        # Select environments
        if env_keys is None:
            env_keys = [k for k, v in self.envs.items()
                        if v.topology in ['network', 'grid']]

        # Select neighbors
        agents = AgentList()
        for key in make_list(env_keys):
            try:
                agents.extend(self.envs[key].neighbors(self))
            except KeyError:
                AgentpyError(f"Agent {self.id} has no environment '{key}'")

        return agents

    def enter(self, env_keys):
        """ Adds the agent to passed environments.

        Arguments:
            env_keys(str or list of str):
                Environments to which the agent should be added.
        """

        try:
            envs = {k: self.model.envs[k] for k in make_list(env_keys)}
        except KeyError as e:
            raise AgentpyError(f'Model has no environment {e}')

        for key, env in envs.items():
            env.agents.append(self)
            self.envs[key] = env

    def exit(self, env_keys=None):
        """ Removes the agent from chosen environments.

        Arguments:
            env_keys(str or list of str, optional):
                Environments from which the agent should be removed.
                If none are given, agent is removed from all its environments.
        """

        if env_keys is None:
            envs = self.envs  # Select all environments by default
        else:
            try:
                envs = {k: self.envs[k] for k in make_list(env_keys)}
            except KeyError as e:
                raise AgentpyError(f'Agent has no environment {e}')

        for key, env in envs.items():
            env.agents.remove(self)
            del self.envs[key]


class AttrList(list):
    """ List of attributes from an :class:`AgentList`.

    Calls are forwarded to each entry and return a list of return values.
    Boolean operators are applied to each entry and return a list of bools.
    Arithmetic operators are applied to each entry and return a new list.
    See :class:`AgentList` for examples.
    """

    def __init__(self, *args, attr=None):
        super().__init__(*args)
        self.attr = attr

    def __repr__(self):
        if self.attr is None:
            return f"AttrList: {list.__repr__(self)}"
        else:
            return f"AttrList of attribute '{self.attr}': " \
                   f"{list.__repr__(self)}"

    def __call__(self, *args, **kwargs):
        return AttrList(
            [func_obj(*args, **kwargs) for func_obj in self],
            attr=self.attr)

    def __eq__(self, other):
        return [obj == other for obj in self]

    def __ne__(self, other):
        return [obj != other for obj in self]

    def __lt__(self, other):
        return [obj < other for obj in self]

    def __le__(self, other):
        return [obj <= other for obj in self]

    def __gt__(self, other):
        return [obj > other for obj in self]

    def __ge__(self, other):
        return [obj >= other for obj in self]

    def __add__(self, v):
        if isinstance(v, AttrList):
            return AttrList([x + y for x, y in zip(self, v)])
        else:
            return AttrList([x + v for x in self])

    def __sub__(self, v):
        if isinstance(v, AttrList):
            return AttrList([x - y for x, y in zip(self, v)])
        else:
            return AttrList([x - v for x in self])

    def __mul__(self, v):
        if isinstance(v, AttrList):
            return AttrList([x * y for x, y in zip(self, v)])
        else:
            return AttrList([x * v for x in self])

    def __truediv__(self, v):
        if isinstance(v, AttrList):
            return AttrList([x / y for x, y in zip(self, v)])
        else:
            return AttrList([x / v for x in self])

    def __iadd__(self, v):
        return self + v

    def __isub__(self, v):
        return self - v

    def __imul__(self, v):
        return self * v

    def __itruediv__(self, v):
        return self / v


class AgentList(list):
    """ List of agents.

    Attribute calls and assignments are applied to all agents
    and return an :class:`AttrList` with attributes of each agent.
    This also works for method calls, which returns a list of return values.
    Arithmetic operators can further be used to manipulate agent attributes,
    and boolean operators can be used to filter list based on agent attributes.

    Examples:

        Let us start by preparing an :class:`AgentList` with three agents::
            
            >>> model = ap.Model()
            >>> model.add_agents(3)
            >>> agents = model.agents
            >>> agents
            AgentList [3 agents]
             
        The assignment operator can be used to set a variable for each agent.
        When the variable is called, an :class:`AttrList` is returned::

            >>> agents.x = 1
            >>> agents.x
            AttrList of attribute 'x': [1, 1, 1]

        One can also set different variables for each agent
        by passing another :class:`AttrList`::

            >>> agents.y = ap.AttrList([1, 2, 3])
            >>> agents.y
            AttrList of attribute 'y': [1, 2, 3]

        Arithmetic operators can be used in a similar way.
        If an :class:`AttrList` is passed, different values are used for
        each agent. Otherwise, the same value is used for all agents::

            >>> agents.x = agents.x + agents.y
            >>> agents.x
            AttrList of attribute 'x': [2, 3, 4]

            >>> agents.x *= 2
            >>> agents.x
            AttrList of attribute 'x': [4, 6, 8]

        Boolean operators can be used to select a subset of agents::

            >>> subset = agents(agents.x > 5)
            >>> subset
            AgentList [2 agents]

            >>> subset.x
            AttrList of attribute 'x': [6, 8]
    """

    def __setattr__(self, name, value):
        if isinstance(value, AttrList):
            # Apply each value to each agent
            assert len(self) == len(value)  # TODO Catch Error
            for obj, v in zip(self, value):
                # print(f"Setting {name} to {v} for {obj}") TODO Remove
                setattr(obj, name, v)
        else:
            # Apply single value to all agents
            for obj in self:
                setattr(obj, name, value)

    def __getattr__(self, name):
        """ Return callable list of attributes """
        return AttrList([getattr(obj, name) for obj in self], attr=name)

    def __repr__(self):
        return f"AgentList [{len(self)} agent{'s' if len(self) > 1 else ''}]"

    def __call__(self, selection):
        return self.select(selection)

    def select(self, selection):
        """ Returns a new :class:`AgentList` based on `selection`.

        Attributes:
            selection (list of bool): List with same length as the agent list.
                Positions that return True will be selected.
        """
        assert len(self) == len(selection)  # TODO Catch Error
        return AgentList([a for a, s in zip(self, selection) if s])

    def random(self, n=1):
        """ Returns a new :class:`AgentList`
        with ``n`` random agents (default 1)."""
        return AgentList(rd.sample(self, n))

    def sort(self, var_key, reverse=False):
        """ Sorts the list based on var_key and returns self """
        super().sort(key=lambda x: x[var_key], reverse=reverse)
        return self

    def shuffle(self):
        """ Shuffles the list and returns self """
        rd.shuffle(self)
        return self


class ApEnv(ApObj):
    """ Agentpy base-class for environments. """

    def __init__(self, model, key):
        super().__init__(model)
        self._agents = AgentList()
        self._topology = None
        self.key = key

    @property
    def topology(self):
        return self._topology

    @property
    def agents(self):
        return self._agents

    def __repr__(self):
        rep = f"Environment '{self.key}'"
        type_ = type(self).__name__
        if type_ != "Environment":
            rep += f" ({type_})"
        return rep

    def __getattr__(self, key):
        raise AgentpyError(f"Environment '{self.key}' has no attribute {key}.")

    def add_agents(self, agents=1, agent_class=Agent, **kwargs):
        """ Adds agents to the environment.

        Arguments:
            agents(int or AgentList, optional): Either number of new agents
                to be created or list of existing agents (default 1).
            agent_class(class, optional): Type of new agents to be created
                if int is passed for agents (default :class:`Agent`).
            **kwargs: Forwarded to :func:`Agent.setup` if new agents are
                created (i.e. if an integer number is passed to `agents`).

        Returns:
            AgentList: List of the new agents.
        """

        # Check if object is environment or model
        is_env = True if self != self.model else False

        # Case 1 - Create new agents
        if isinstance(agents, int):
            agents = AgentList([agent_class(self.model, **kwargs)
                                for _ in range(agents)])
            if is_env:  # Add agents to master list
                self.model._agents.extend(agents)

        # Case 2 - Add existing agents
        else:
            if not isinstance(agents, AgentList):
                agents = AgentList(agents)

        # Add environment to agents
        if is_env:
            for agent in agents:
                agent.envs[self.key] = self

        # Add agents to environment
        self._agents.extend(agents)

        return agents


class Environment(ApEnv):
    """ Standard environment for agents (no topology).

    This class can be used as a parent class for custom environment types.

    All agentpy model objects call the method ``setup()`` after creation,
    can access class attributes like dictionary items,
    and can be removed from the model with the ``del`` statement.

    Attributes:
        model (Model): The model instance.
        agents (AgentList): The environments' agents.
        p (AttrDict): The models' parameters.
        key (str): The environments' name.
        topology (str): Topology of the environment.
        log (dict): The environments' recorded variables.

    Arguments:
        model (Model): The model instance.
        key (str, optional): The environments' name.
        **kwargs: Will be forwarded to :func:`Environment.setup`.
    """

    def __init__(self, model, key, **kwargs):
        super().__init__(model, key)
        self._set_var_ignore()
        self.setup(**kwargs)


class Network(ApEnv):
    """ Agent environment with a graph topology.

    This class can be used as a parent class for custom network types.
    Unknown attribute and method calls are forwarded to self.graph.

    Notes:
        All agentpy model objects can access attributes as items
        and will call the method ``setup()`` after creation (if defined).

    Arguments:
        model (Model): The model instance.
        key (str, optional): The environments' name.
        graph (networkx.Graph): The environments' graph.
        **kwargs: Will be forwarded to :func:`Network.setup`.
    """

    def __init__(self, model, key, graph=None, **kwargs):

        super().__init__(model, key)

        if graph is None:
            self.graph = nx.Graph()
        elif isinstance(graph, nx.Graph):
            self.graph = graph
        else:
            raise TypeError("'graph' must be of type networkx.Graph")

        self._topology = 'network'
        self._set_var_ignore()
        self.setup(**kwargs)

    def add_agents(self, agents, agent_class=Agent,  # noqa
                   map_to_nodes=False, **kwargs):
        """ Adds agents to the network environment.
        See :func:`Environment.add_agents` for standard arguments.
        Additional arguments for the network are listed below.

        Arguments:
            map_to_nodes(bool,optional): Map new agents to each node of the
                graph (default False). Should be used if a graph with empty
                nodes has been passed at network creation.
        """

        # Standard adding
        new_agents = super().add_agents(agents, agent_class, **kwargs)

        # Extra network features
        if map_to_nodes:
            # Map each agent to a node of the graph
            if len(new_agents) != len(self.graph.nodes):
                raise AgentpyError(
                    f"Number of agents ({len(new_agents)}) does not "
                    f"match number of nodes ({len(self.graph.nodes)})")
            mapping = {i: agent for i, agent in enumerate(new_agents)}
            nx.relabel_nodes(self.graph, mapping=mapping, copy=False)
        else:
            # Add agents to graph as new nodes
            for agent in new_agents:
                self.graph.add_node(agent)

    def neighbors(self, agent):
        """ Returns an :class:`AgentList` of agents
        that are connected to the passed agent. """
        return AgentList([n for n in self.graph.neighbors(agent)])

    def __getattr__(self, name):
        # Forward unknown method call to self.graph
        def method(*args, **kwargs):
            return getattr(self.graph, name)(*args, **kwargs)

        try:
            return method
        except AttributeError:
            raise AttributeError(
                f"Environment '{self.key}' has no attribute '{name}'")


class Grid(ApEnv):
    """ Grid environment that contains agents with a spatial topology.
    Inherits attributes and methods from :class:`Environment`.

    Attributes:
        _positions(dict): Agent positions.

    Arguments:
        model (Model): The model instance.
        key (dict or EnvDict, optional):  The environments' name
        dim(int, optional): Number of dimensions (default 2).
        size(int or tuple): Size of the grid.
            If int, the same length is assigned to each dimension.
            If tuple, one int item is required per dimension.
        **kwargs: Will be forwarded to :func:`Grid.setup`.
    """

    def __init__(self, model, key, shape, **kwargs):

        super().__init__(model, key)

        self._topology = 'grid'
        self._grid = make_matrix(make_list(shape), AgentList)
        self._positions = {}
        self._shape = shape
        self._set_var_ignore()
        self.setup(**kwargs)

    @property
    def grid(self):
        return self._grid

    @property
    def shape(self):
        return self._shape

    def neighbors(self, agent, diagonal=False):

        """ Returns agent neighbors. """

        if diagonal:  # Include diagonal neighbors (square shape)
            return self._get_neighbors8(self._positions[agent], self._grid)
        else:  # Diamond shape
            return self._get_neighbors4(self._positions[agent], self._grid)

    def area(self, area):

        return AgentList(self._get_area(area, self._grid))

    def _get_area(self, area, grid):

        """ Returns agents in area of style:
        [(x_min,x_max),(y_min,y_max),...]"""

        subgrid = grid[area[0][0]:area[0][1] + 1]

        # Detect last row (must have AgentLists)
        if isinstance(subgrid[0], AgentList):
            # Flatten list of AgentLists to list of agents
            return [y for x in subgrid for y in x]

        objects = []
        for row in subgrid:
            objects.extend(self._get_area(area[1:], row))

        return objects

    def _get_neighbors4(self, pos, grid, dist=1):
        """ Return agents in diamond-shaped area around pos."""
        subgrid = grid[max(0, pos[0] - dist):pos[0] + dist + 1]

        if len(pos) == 1:
            return [y for x in subgrid for y in x]  # flatten list

        objects = []
        for row, dist in zip(subgrid, [0, 1, 0]):
            objects.extend(self._get_neighbors4(pos[1:], row, dist))

        return objects

    def _get_neighbors8(self, pos, grid):
        """ Return agents in square-shaped area around pos."""
        subgrid = grid[max(0, pos[0] - 1):pos[0] + 2]

        if len(pos) == 1:
            return [y for x in subgrid for y in x]  # flatten list

        objects = []
        for row in subgrid:
            objects.extend(self._get_neighbors8(pos[1:], row))

        return objects

    def _get_pos(self, grid, pos):
        if len(pos) == 1:
            return grid[pos[0]]
        return self._get_pos(grid[pos[0]], pos[1:])

    def get_pos(self, pos):
        return self._get_pos(self._grid, pos)

    def change_pos(self, agent, new_position):
        self.get_pos(self._positions[agent]).drop(agent)  # Remove old position
        self.get_pos(new_position).append(agent)  # Add new position (grid)
        self._positions[agent] = new_position  # Add new position (dict)

    def add_agents(self, agents, agent_class=Agent, positions=None,
                   random=False, map_to_grid=False, **kwargs):
        """ Adds agents to the grid environment.
        See :func:`Environment.add_agents` for standard arguments."""

        # TODO unfinished

        """Additional arguments for the grid environment are listed below.

        Arguments:
            map_to_grid(bool, optional): Map new agents to each position
             in the grid (default False). Should be used if a graph with empty
                nodes has been passed at network creation."""

        # Standard adding
        new_agents = super().add_agents(agents, agent_class, **kwargs)

        # Extra grid features
        if map_to_grid:
            pass  # TODO unfinished
        elif positions:
            for agent, pos in zip(new_agents, positions):
                self._positions[agent] = pos
        elif random:
            positions = list(product(*[range(i) for i in self.shape]))
            sample = rd.sample(positions, agents)

            for agent, pos in zip(new_agents, sample):
                self._positions[agent] = pos  # Log Position
                self.get_pos(pos).append(agent)  # Add to new position

        else:
            for agent in new_agents:
                self._positions[agent] = [0] * self.dim


class EnvDict(dict):
    """ Dictionary for environments """

    def __repr__(self):

        return f"EnvDict {{{len(self.keys())} environments}}"

    def do(self, method, *args, **kwargs):
        """ Calls ``method(*args,**kwargs)``
        for all environments in the dictionary. """

        for env in self.values():
            getattr(env, method)(*args, **kwargs)

    def select(self, selection):
        """ Returns a new :class:`EnvDict` based on `selection`.

        Attributes:
            selection (list of str): List of keys to be included.
        """  # TODO NEW TEST

        return EnvDict({k: v for k, v in self.envs.items()
                        if k in make_list(selection)})

    def add_agents(self, agents=1, agent_class=Agent, **kwargs):
        """ Adds agents to all environments.
        See :func:`Environment.add_agents`"""

        envs = list(self.values())
        new_agents = envs[0].add_agents(agents, agent_class, **kwargs)

        if len(envs) > 0:
            for env in envs[1:]:
                env.add_agents(new_agents)


class Model(ApEnv):
    """
    An agent-based model that can hold environments and agents.

    This class can be used as a parent class for custom models.
    Attributes can be accessed as items, and
    environments can be accessed as attributes.
    See :func:`Model.run` for information on the simulation procedure.

    Attributes:
        name (str): The models' name.
        envs (EnvDict): The models' environments.
        agents (AgentList): The models' agents.
        p (AttrDict): The models' parameters.
        t (int): Current time-step of the model.
        t_max (int): Time limit for simulations (default 1_000_000).
        log (dict): The models' recorded variables.
        output (DataDict): Output data after simulation.

    Arguments:
        parameters (dict, optional): Dictionary of model parameters.
            Recommended types for parameters are int, float, str, list,
            numpy.integer, numpy.floating, and numpy.ndarray.
        run_id (int, optional): Number of current run.
        scenario (str, optional): Current scenario.
    """

    def __init__(self,
                 parameters=None,
                 run_id=None,
                 scenario=None):

        super().__init__(self, 'model')

        self.t = 0
        self.t_max = 1_000_000
        self.name = type(self).__name__
        self.run_id = run_id
        self.scenario = scenario

        # Recording
        self.measure_log = {}
        self.output = DataDict()
        self.output.log = {'name': self.name,
                           'time_stamp': str(datetime.now())}

        # Private variables
        self._parameters = AttrDict(parameters)
        self._stop = False
        self._id_counter = -1
        self._set_var_ignore()

    def __repr__(self):

        rep = "Agent-based model {"
        ignore = ['measure_log', 'key', 'output']

        for prop in ['agents', 'envs']:
            rep += f"\n'{prop}': {self[prop]}"

        for k, v in self.__dict__.items():

            if k not in ignore and not k[0] == '_':
                rep += f"\n'{k}': {v}"

            if k == 'output':
                rep += f"\n'{k}': DataDict with {len(v.keys())} entries"

        return rep + ' }'

    def __getattr__(self, key):
        try:  # Try to access environments
            return self.envs[key]
        except KeyError:
            raise AttributeError(
                f"Model has no attribute or environment '{key}'")

    @property
    def objects(self):
        """The models agents and environments (list of objects)."""
        return self.agents + list(self.envs.values())

    def _new_id(self):

        self._id_counter += 1
        return self._id_counter

    def add_env(self, env_key, env_class=Environment, **kwargs):

        """ Creates a new environment. """

        for env_key in make_list(env_key):
            self.envs[env_key] = env_class(self.model, env_key, **kwargs)

    def add_network(self, env_key, env_class=Network, **kwargs):

        """ Creates a new network environment. """

        for env_key in make_list(env_key):
            self.envs[env_key] = env_class(self.model, env_key, **kwargs)

    def add_grid(self, env_key, env_class=Grid, **kwargs):

        """ Creates a new spacial grid environment. """

        for env_key in make_list(env_key):
            self.envs[env_key] = env_class(self.model, env_key, **kwargs)

    def measure(self, measure, value):

        """ Records an evaluation measure. """

        self.measure_log[measure] = [value]

    # Main simulation functions

    def setup(self):
        """ Defines the model's actions before the first simulation step.
        Can be overwritten and used to initiate agents and environments."""

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

    def _update_stop(self, steps):
        """ Stops :meth:`model.run` if steps or t_max is reached. """

        if steps is not False and self.t >= steps:
            self.stop()

        if self.t >= self.t_max:
            raise AgentpyError(
                f"Time limit 't_max == {self.t_max}' has been reached."
                "You can set a higher limit by adjusting ``Model.t_max``")

    def run(self, display=True):
        """ Executes the simulation of the model.

        The simulation proceeds as follows.
        It starts by calling :func:`Model.setup` and :func:`Model.update`.
        After that, ``Model.t`` is increased by 1 and
        :func:`Model.step` and :func:`Model.update` are called.
        This step is repeated until the method :func:`Model.stop` is called
        or the parameter steps has been reached (``Model.t >= Model.p.steps``).
        After the last step, :func:`Model.end` is called.

        Arguments:
            display(bool,optional):
                Whether to display simulation progress (default True).

        Returns:
            DataDict: Recorded model data,
                which can also be found in ``Model.output``.

        Raises:
            AgentpyError: If simulation time reaches maximum
                (``Model.t >= Model.t_max``).

        """

        dt0 = datetime.now()  # Time-Stamp
        steps = self.p['steps'] if 'steps' in self.p else False

        self._stop = False
        self.setup()
        self.update()
        self._update_stop(steps)

        while not self._stop:

            self.t += 1
            self.step()
            self.update()
            self._update_stop(steps)

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

        def output_from_obj_list(self, obj_list, id_or_key, columns):
            # Aggregate logs per object type
            obj_types = {}
            for obj in obj_list:

                if obj.log:  # Check for variables

                    # Add object id/key to object log
                    obj.log['obj_id'] = [obj[id_or_key]] * len(obj.log['t'])

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
        if self.p is not None:
            self.output['parameters'] = self.p

        # 1 - Define additional index columns
        columns = {}
        if self.run_id is not None:
            columns['run_id'] = self.run_id
        if self.scenario is not None:
            columns['scenario'] = self.scenario

        # 2 - Create measure output
        if self.measure_log:
            d = self.measure_log
            for key, value in columns.items():
                d[key] = value
            df = pd.DataFrame(d)
            if columns:
                df = df.set_index(list(columns.keys()))
            self.output['measures'] = df

        # 3 - Create variable output
        self.output['variables'] = DataDict()

        # 3.1 - Create variable output for objects
        output_from_obj_list(self, self.agents, 'id', columns)
        output_from_obj_list(self, self.envs.values(), 'key', columns)

        # 3.2 - Create variable output for model
        if self.log:
            df = pd.DataFrame(self.log)
            # df['obj_id'] = 'model'
            for k, v in columns.items():
                df[k] = v
            df = df.set_index(list(columns.keys()) + ['t'])  # 'obj_id',

            if self.output['variables']:
                self.output['variables']['model'] = df
            else:
                self.output['variables'] = df  # No subdict if only model vars

        # 3.3 - Remove variable dict if empty (i.e. nothing has been added)
        elif not self.output['variables']:
            del self.output['variables']
