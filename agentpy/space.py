"""
Agentpy Space Module
Content: Class for continuous spatial environments
"""

# There is much room for optimization in this module.
# Contributions are welcome! :)

# TODO Add option of space without shape (infinite)
# TODO Improve datatype consistency (list, np.array, etc.)
# TODO Method for distance between agents
# TODO Create method items() consistent with Grid

import itertools
import numpy as np
import random as rd
import collections.abc as abc
from scipy import spatial
from .objects import ApEnv, Agent
from .tools import make_list, make_matrix
from .lists import AgentList


class Space(ApEnv):
    """ Environment that contains agents with a continuous spatial topology.
    To add new space environments to a model, use :func:`Model.add_space`.
    For a discrete spatial topology, use :class:`Grid`.

    This class can be used as a parent class for custom space types.
    All agentpy model objects call the method :func:`setup` after creation,
    and can access class attributes like dictionary items.
    See :class:`Environment` for general properties of all environments.

    Arguments:
        model (Model): The model instance.
        shape (tuple of float): Size of the space.
            The length of the tuple defines the number of dimensions,
            and the values in the tuple define the length of each dimension.
        connect_borders (bool, optional):
            Whether to connect borders (default False).
            If True, space will be toroidal, meaning that agents who, for
            example, move over the right border, will appear on the left side.
        **kwargs: Will be forwarded to :func:`Space.setup`.

    Attributes:
        shape (tuple of float): Length of each spatial dimension.
        dim (int): Number of dimensions.
    """

    def __init__(self, model, shape, connect_borders=False, **kwargs):

        super().__init__(model)

        self._topology = 'space'
        self._connect_borders = connect_borders
        self._cKDTree = None
        self._sorted_agents = None
        self._sorted_agent_points = None
        self._agent_dict = {}  # agent instance : agent position (np.array)
        self._shape = tuple(shape)
        self._set_var_ignore()
        self.setup(**kwargs)

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return len(self._shape)

    @property
    def KDTree(self):
        """ KDTree of agent positions for neighbor lookup,
        using :class:`scipy.spatial.KDTree`.
        Tree is recalculated if agent's have moved or changed.
        If there are no agents, returns None."""
        # Create new KDTree if necessary
        if self._cKDTree is None and len(self.agents) > 0:
            self._sorted_agents = []
            self._sorted_agent_points = []
            for a in self.agents:
                self._sorted_agents.append(a)
                self._sorted_agent_points.append(self._agent_dict[a])
            if self._connect_borders:
                self._cKDTree = spatial.cKDTree(self._sorted_agent_points,
                                                boxsize=self.shape)
            else:
                self._cKDTree = spatial.cKDTree(self._sorted_agent_points)
        return self._cKDTree  # Return existing or new KDTree

    def add_agents(self, agents=1, agent_class=Agent, positions=None,  # noqa
                   random=False, generator=None, **kwargs):
        """ Adds agents to the space environment, and returns new agents.
        See :func:`Environment.add_agents` for standard arguments.
        Additional arguments are listed below.

        Arguments:
            positions (array_like, optional):
                The positions of the added agents.
                Array must have the same length as number of agents
                to be added, and each entry must be an array with coordinates.
                If none is passed, agents will be placed in the
                bottom-left corner, i.e.: (0, 0, ...).
            random (bool, optional):
                If no positions are passed, agents will be placed in random
                locations instead of starting at the corner (default False).
            generator (numpy.random.Generator, optional):
                Random number generator.
                If none is passed, :obj:`Model.random` is used.
        """

        # Standard adding
        new_agents = super().add_agents(agents, agent_class, **kwargs)

        # Extra space features
        if not positions:
            n_agents = len(new_agents)
            if random:
                generator = generator if generator else self.model.random
                positions = [[generator.random() * d_max
                              for d_max in self.shape]
                             for _ in range(n_agents)]
            else:
                positions = [np.zeros(self.dim) for _ in range(n_agents)]

        for agent, pos in zip(new_agents, positions):
            self._agent_dict[agent] = np.array(pos)  # Add pos to agent_dict

        return new_agents

    def remove_agents(self, agents):
        """ Removes agents from the environment. """
        for agent in make_list(agents):
            del self._agent_dict[agent]
        super().remove_agents(agents)

    def position(self, agent):
        """ Returns :class:`numpy.array` with position of passed agent.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
        """
        if isinstance(agent, int):
            agent = self.model.get_obj(agent)
        return self._agent_dict[agent]

    def positions(self, transpose=False):
        """ Returns list with positions of all agents.

        Arguments:
            transpose (bool, optional):
                If False (default), positions will be of style:
                [[agent1.x, agent1.y, ...], [agent2.x, ...], ...]
                If True, positions will be of style:
                [[agent1.x, agent2.x, ...], [agent1.y, ...], ...]
        """
        pos_per_agent = self._agent_dict.values()
        if not transpose:
            return pos_per_agent
        else:
            return map(list, zip(*pos_per_agent))

    def get_agents(self, center, radius):
        """ Returns an :class:`AgentList` with agents in selected search area,
        using :func:`scipy.cKDTree.query_ball_point`.

        Arguments:
            center (array_like): Coordinates of the center of the search area.
            radius (float): Radius around the center in which to search.
        """
        if self.KDTree:
            list_ids = self.KDTree.query_ball_point(center, radius)
            agents = [self._sorted_agents[list_id] for list_id in list_ids]
            return AgentList(agents, self.model)
        else:
            return AgentList(model=self.model)

    def neighbors(self, agent, distance):
        """ Returns an :class:`AgentList` with agent neighbors,
        using :func:`scipy.cKDTree.query_ball_point`.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
            distance(float): Radius around the agent in which to
                search for neighbors.
        """

        list_ids = self.KDTree.query_ball_point(
            self._agent_dict[agent], distance)
        agents = [self._sorted_agents[list_id] for list_id in list_ids]
        agents.remove(agent)  # Remove original agent
        return AgentList(agents, self.model)

    def move_agent(self, agent, position):
        """ Moves agent to new position.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
            position(array_like): New position of the agent.
        """

        if isinstance(agent, int):
            agent = self.model.get_obj(agent)

        for i in range(len(position)):

            # Border behavior
            if self._connect_borders:  # Connected
                while position[i] > self.shape[i]:
                    position[i] -= self.shape[i]
                while position[i] < 0:
                    position[i] += self.shape[i]
            else:  # Not connected - Stop at border
                if position[i] > self.shape[i]:
                    position[i] = self.shape[i]
                elif position[i] < 0:
                    position[i] = 0

        # Updating position
        self._agent_dict[agent] = np.array(position)
        self._cKDTree = None  # Reset KDTree
