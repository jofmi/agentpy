"""
Agentpy Objects Module
Content: Classes for environments and agents
"""

from .lists import AgentList, EnvList
from .tools import AgentpyError, make_list


class ApObj:
    """ Agentpy base-class for objects of agent-based models."""

    def __init__(self, model):
        self._log = {}
        self._model = model
        self._envs = EnvList()
        self._var_ignore = []
        self._id = model._new_id()  # Assign id to new object
        self._model._obj_dict[self.id] = self  # Add object to object dict

    def __repr__(self):
        return f"{self.type} (Obj {self.id})"

    def __getattr__(self, key):
        raise AttributeError(f"{self} has no attribute '{key}'.")

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
    def id(self):
        return self._id

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
    def env(self):
        """ The objects first environment. """
        return self._envs[0] if self._envs else None

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
    and can access class attributes like dictionary items. To add new agents
    to a model, use :func:`Model.add_agents` or :func:`Environment.add_agents`.

    Arguments:
        model (Model): Instance of the current model.
        **kwargs: Will be forwarded to :func:`Agent.setup`.

    Attributes:
        model (Model): Model instance.
        p (AttrDict): Model parameters.
        envs (EnvList): Environments of the agent.
        log (dict): Recorded variables of the agent.
        id (int): Unique identifier of the agent instance.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model)
        self._set_var_ignore()
        self.setup(**kwargs)

    def delete(self):
        """ Remove agent from all environments and the model. """
        for env in self.envs:
            env.remove_agents(self)
        self.model.remove_agents(self)

    def _find_env(self, env=None, topologies=None, new=False):
        """ Return obj of id or first object with topology. """
        # TODO Select based on method existance instead of topology
        if topologies:
            return self._find_env_top(env=env, topologies=topologies)
        if env is None:
            if self.envs:
                return self.envs[0]
            raise AgentpyError(f"{self} has no environments.")
        else:
            if isinstance(env, int):
                env = self.model.get_obj(env)
            if new or env in self.envs:
                return env
            raise AgentpyError(f"{self} is not part of environment {env}.")

    def _find_env_top(self, env=None, topologies=None):
        """ Return obj of id or first object with topology"""
        topologies = make_list(topologies)
        if env is None:
            for env in self.envs:
                if env.topology in topologies:
                    return env
            raise AgentpyError(
                f"Agent {self.id} has no environment "
                f"with topology '{topologies}'")
        else:
            if isinstance(env, int):
                env = self.model.get_obj(env)
            if hasattr(env, 'topology') and env.topology in topologies:
                return env
            raise AgentpyError(
                f"{env} does not have topology '{topologies}'")

    def position(self, env=None):
        """ Returns the agents' position from a grid.

        Arguments:
            env (int or Environment, optional):
                Instance or id of environment that should be used.
                Must have topology 'grid'.
                If none is given, the first environment of that topology
                in :attr:`Agent.envs` is used.
        """

        env = self._find_env(env, ['grid', 'space'])
        return env._agent_dict[self]

    def move_by(self, path, env=None):
        """ Changes the agents' location in the selected environment,
        relative to the current position.

        Arguments:
            path (list of int): Relative change of position.
            env (int or Environment, optional):
                Instance or id of environment that should be used.
                Must have topology 'grid'.
                If none is given, the first environment of that topology
                in :attr:`Agent.envs` is used.
        """
        # TODO Add border jumping feature (toroidal)
        env = self._find_env(env, ['grid', 'space'])
        old_pos = self.position(env)
        position = [p + c for p, c in zip(old_pos, path)]
        env.move_agent(self, position)

    def move_to(self, position, env=None):
        """ Changes the agents' location in the selected environment.

        Arguments:
            position (list of int): Position to move to.
            env(int or Environment, optional):
                Instance or id of environment that should be used.
                Must have topology 'grid'.
                If none is given, the first environment of that topology
                in :attr:`Agent.envs` is used.
        """

        env = self._find_env(env, ['grid', 'space'])
        env.move_agent(self, position)

    def neighbors(self, env=None, distance=1, **kwargs):
        """ Returns the agents' neighbor's from an environment,
        by calling the environments :func:`neighbors` function.

        Arguments:
            env(int or Environment, optional):
                Instance or id of environment that should be used.
                Must have topology 'grid' or 'network'.
                If none is given, the first environment of that topology
                in :attr:`Agent.envs` is used.
            distance(int, optional):
                Distance from agent in which to look for neighbors.
            **kwargs: Forwarded to the environments :func:`neighbors` function.

        Returns:
            AgentList: Neighbors of the agent.
        """
        env = self._find_env(env, ('grid', 'space', 'network'))
        return env.neighbors(self, distance=distance, **kwargs)

    def enter(self, env):
        """ Adds agent to passed environment.

        Arguments:
            env(int or Environment, optional):
                Instance or id of environment that should be used.
                If none is given, the first environment
                in :attr:`Agent.envs` is used.
        """
        env = self._find_env(env, new=True)
        env.add_agents(self)

    def exit(self, env=None):
        """ Removes agent from chosen environment.

        Arguments:
            env(int or Environment, optional):
                Instance or id of environment that should be used.
                If none is given, the first environment
                in :attr:`Agent.envs` is used.
        """
        env = self._find_env(env)
        env.remove_agents(self)


class ApEnv(ApObj):
    """ Agentpy base-class for environments. """

    def __init__(self, model):
        super().__init__(model)
        self._agents = AgentList(model=model)
        self._topology = None

    @property
    def topology(self):
        return self._topology

    @property
    def agents(self):
        return self._agents

    def remove_agents(self, agents):
        """ Removes agents from the environment. """
        is_env = True if self != self.model else False
        for agent in list(make_list(agents)):  # Soft copy
            self._agents.remove(agent)
            if is_env:
                agent.envs.remove(self)

    def add_agents(self, agents=1, agent_class=Agent, **kwargs):
        """ Adds agents to the environment.

        Arguments:
            agents(int or AgentList, optional): Either number of new agents
                to be created or list of existing agents (default 1).
            agent_class(type, optional): Type of new agents to be created
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
                                for _ in range(agents)], model=self.model)
            if is_env:  # Add agents to master list
                self.model._agents.extend(agents)

        # Case 2 - Add existing agents
        else:
            if not isinstance(agents, AgentList):
                agents = AgentList(make_list(agents), model=self.model)

        # Add environment to agents
        if is_env:
            for agent in agents:
                agent.envs.append(self)

        # Add agents to environment
        self._agents.extend(agents)

        return agents


class Environment(ApEnv):
    """ Standard environment for agents (no topology).

    This class can be used as a parent class for custom environment types.
    All agentpy model objects call the method :func:`setup()` after creation,
    and can access class attributes like dictionary items. To add new
    environments to a model, use :func:`Model.add_env`.

    Arguments:
        model (Model): The model instance.
        **kwargs: Will be forwarded to :func:`Environment.setup`.

    Attributes:
        model (Model): Model instance.
        p (AttrDict): Model parameters.
        id (int): Unique identifier of the environment instance.
        topology (str): Topology of the environment.
        log (dict): The environments' recorded variables.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model)
        self._set_var_ignore()
        self.setup(**kwargs)
