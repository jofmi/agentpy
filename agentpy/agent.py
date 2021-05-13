"""
Agentpy Agent Module
Content: Agent Classes
"""

from .objects import Object
from .sequences import AgentList
from .tools import AgentpyError, make_list


class Agent(Object):
    """ Template for an individual agent.

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
                f"{self} is already part of {self.env}. Consider using "
                "'MultiAgent' for agents with multiple environments.")
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
                Type depends on the environment.
        """
        new_pos = [p + c for p, c in zip(self.pos, path)]
        self.env.move_agent(self, new_pos)

    def move_to(self, pos):
        """ Changes the agent's absolute position in its environment.

        Arguments:
            pos: Position or node to move to.
                Type depends on the environment.
        """
        self.env.move_agent(self, pos)

    def neighbors(self, distance=1):
        """ Returns the agent's neighbors from its environment.

        Arguments:
            distance(int, optional):
                Distance in which to look for neighbors (default 1).
                At the moment, values other than `1` are not supported for
                environments of type :class:`Network`.

        Returns:
            AgentList: Neighbors of the agent.
        """
        # TODO Remove network notice when fixed
        return self.env.neighbors(self, distance=distance)


class MultiAgent(Agent):
    """ Template for an individual agent
    that can be part of multiple environments.
    Attributes and methods are the same as :class:`Agent`
    except for ones described here.

    Attributes:
        env (dict): Dictionary connecting each of the agent's environments
            to the agent's position in that environment.
        pos (dict): Equivalent to `env`.
    """

    def __init__(self, model, *args, **kwargs):
        Object.__init__(self, model)
        self.env = self.pos = {}
        self.setup(*args, **kwargs)

    # Environment change ---------------------------------------------------- #

    def _add_env(self, env, pos=None):
        self.env[env] = pos

    def _remove_env(self, env):
        del self.env[env]

    # Environment access ---------------------------------------------------- #

    def move_by(self, env, path):
        """ Changes the agent's location in the selected environment,
        relative to the current position.

        Arguments:
            env (Grid or Space):
                Instance of a spatial environment that the agent is part of.
            path: Relative change of position.
                Type depends on the environment.

        """
        old_pos = self.pos[env]
        new_pos = [p + c for p, c in zip(old_pos, path)]
        env.move_agent(self, new_pos)

    def move_to(self, env, pos):
        """ Changes the agent's location in the selected environment.

        Arguments:
            env (Grid or Space or Network):
                Instance of a spatial environment that the agent is part of.
            pos: Position to move to.
                Type depends on the environment.
        """
        env.move_agent(self, pos)

    def neighbors(self, env, distance=1):
        """ Returns the agent's neighbors from the selected environment.

        Arguments:
            env (Grid or Space or Network):
                Instance of environment that should be used.
            distance (int, optional):
                Distance from agent in which to look for neighbors (default 1).
                At the moment, values other than `1` are not supported for
                environments of type :class:`Network`.

        Returns:
            AgentList: Neighbors of the agent.
        """
        # TODO Remove network notice when fixed
        return env.neighbors(self, distance=distance)

