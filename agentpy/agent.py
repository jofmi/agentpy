"""
Agentpy Agent Module
Content: Agent Classes
"""

from .objects import Object
from .sequences import AgentList
from .tools import AgentpyError, make_list


class Agent(Object):
    """ Template for an individual agent
    that can be part of zero or one environments.

    Arguments:
        model (Model): The model instance.
        **kwargs: Will be forwarded to :func:`Agent.setup`.

    Attributes:
        id (int): Unique identifier of the agent.
        log (dict): Recorded variables of the agent.
        type (str): Class name of the agent.
        model (Model): The model instance.
        p (AttrDict): The model parameters.
        env (Grid, Space, or Network): Environment of the agent.
        pos (Position or AgentNode): Position of the agent in its environment.
        vars (list of str): Names of the agent's custom variables.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        self.env = None
        self.pos = None
        self.setup(*args, **kwargs)

    # Environment change ---------------------------------------------------- #

    def _add_env(self, env, pos=None):

        if self.env is not None:
            raise AgentpyError(
                f"{self} is already part of an environment."
                "Use 'agentpy.MultiAgent' for multi-environment agents.")
        self.env = env
        self.pos = pos

    def _remove_env(self, env):

        if self.env is not env:
            raise AgentpyError(f"Agent is not part of {env}.")
        self.env = None
        self.pos = None

    # Environment access ---------------------------------------------------- #

    def move_by(self, path):
        """ Changes the agent's relative position in its environment,

        Arguments:
            path: Relative change of position.
                Type and structure depend on the environment.

        Returns:
            position: The new position
        """
        new_pos = [p + c for p, c in zip(self.pos, path)]
        self.env.move_agent(self, new_pos)
        return new_pos

    def move_to(self, pos):
        """ Changes the agent's absolute position in its environment.

        Arguments:
            pos: Position or node to move to.
                Type and structure depend on the environment.
        """
        self.env.move_agent(self, pos)

    def neighbors(self, distance=1, **kwargs):
        """ Returns the agent's neighbors from its environment.

        Arguments:
            distance(int, optional):
                Distance in which to look for neighbors (default 1).
            **kwargs:
                Forwarded to the environments :func:`neighbors` function.

        Returns:
            AgentList: Neighbors of the agent.
        """
        return self.env.neighbors(self, distance=distance, **kwargs)


class MultiAgent(Agent):
    """ Template for an individual agent
    that can be part of multiple environments.
    Attributes and methods are inherited from :class:`Agent`,
    except for the differences described here.

    Attributes:
        env (iterator): Iterator over the agent's environments.
        pos (dict): Dictionary connecting each of the agent's environments
            to the agent's position in that environment.

    To Do:
        This class is still in development.
    """

    # Delete env / pos

    def _add_env(self, env, pos=None):

        self.envs[env] = pos

    def _remove_env(self, env):

        del self.envs[env]

    def move(self, path, env=None):
        """ Changes the agents' location in the selected environment,
        relative to the current position.

        Arguments:
            path:
                Relative change of position.
                Type and structure depend on the environment.
            env (Environment, optional):
                Instance of a spatial environment
                that the agent is part of.
                If the agent has only one environment,
                it is selected by default.
        """
        if env is None:
            env = self.env
            if isinstance(self.env, list):
                raise AgentpyError(f"{self} has more than one environment."
                                   "'Agent.move_by' needs an argument 'env'.")
        old_pos = self.pos  # ition(env)
        new_pos = [p + c for p, c in zip(old_pos, path)]
        env.move_agent(self, new_pos)

    def move_to(self, pos, env=None):
        """ Changes the agents' location in the selected environment.

        Arguments:
            pos:
                Position to move to.
                Type and structure depend on the environment.
            env (int or Environment, optional):
                Instance or id of a spatial environment
                that the agent is part of.
                If the agent has only one environment,
                it is selected by default.
        """

        if env is None:
            env = self.env
            if isinstance(self.env, list):
                raise AgentpyError(f"{self} has more than one environment."
                                   "'Agent.move_to' needs an argument 'env'.")
        env.move_agent(self, pos)

    def neighbors(self, env=None, distance=1, **kwargs):
        """ Returns the agents' neighbors from its environments.

        Arguments:
            env(int or Environment or list, optional):
                Instance or id of environment that should be used,
                or a list with multiple instances and/or ids.
                If none are given, all of the agents environments are used.
            distance(int, optional):
                Distance from agent in which to look for neighbors.
            **kwargs: Forwarded to the environments :func:`neighbors` function.

        Returns:
            AgentList: Neighbors of the agent. Agents without environments
            and environments without a topology like :class:`Environment`
            will return an empty list. If an agent has the same neighbors
            in multiple environments, duplicates will be removed.
        """
        #if env:
        #    if isinstance(env, (list, tuple)):
        #        envs = [self._find_env(en) for en in env]
        #    else:
        #        return self._find_env(env).neighbors(
        #            self, distance=distance, **kwargs)
        #elif len(self.envs) == 0:
        #    return AgentList()
        #elif len(self.envs) == 1:
        #    return self.envs[0].neighbors(self, distance=distance, **kwargs)
        #else:
        #    envs = self.envs

        agents = AgentList()

        if env is None:
            env = make_list(self.env)

        for e in env:
            agents.extend(e.neighbors(self, distance=distance, **kwargs))

        return agents

        #agents = AgentList()
        # TODO
        #for env in envs:
        #    agents.extend(env.neighbors(self, distance=distance, **kwargs))

        ## TODO Better way to remove duplicates?
        #return AgentList(dict.fromkeys(agents))
