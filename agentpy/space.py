"""
Agentpy Space Module
Content: Class for continuous spatial environments
"""

# TODO Add option of space without shape (infinite)
# TODO Custom iterator for neighbors() & select() for performance

import itertools
import numpy as np
import random as rd
import collections.abc as abc
from scipy import spatial
from .objects import Object
from .tools import make_list, make_matrix
from .sequences import AgentList, AgentIter


class Space(Object):
    """ Environment that contains agents with a continuous spatial topology.
    To add new space environments to a model, use :func:`Model.add_space`.
    For a discrete spatial topology, see :class:`Grid`.

    This class can be used as a parent class for custom space types.
    All agentpy model objects call the method :func:`setup` after creation,
    and can access class attributes like dictionary items.

    Arguments:
        model (Model): The model instance.
        shape (tuple of float): Size of the space.
            The length of the tuple defines the number of dimensions,
            and the values in the tuple define the length of each dimension.
        torus (bool, optional):
            Whether to connect borders (default False).
            If True, the space will be toroidal, meaning that agents who
            move over a border will re-appear on the opposite side.
            If False, they will remain at the edge of the border.
        **kwargs: Will be forwarded to :func:`Space.setup`.

    Attributes:
        agents (AgentIter):
            Iterator over all agents in the space.
        positions (dict of Agent):
            Dictionary linking each agent instance to its position.
        shape (tuple of float):
            Length of each spatial dimension.
        ndim (int):
            Number of dimensions.
        kdtree (scipy.spatial.cKDTree or None):
            KDTree of agent positions for neighbor lookup.
            Will be recalculated if agents have moved.
            If there are no agents, tree is None.
    """

    def __init__(self, model, shape, torus=False, **kwargs):

        super().__init__(model)

        self._torus = torus
        self._cKDTree = None
        self._sorted_agents = None
        self._sorted_agent_points = None

        self.positions = {}
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

        self._set_var_ignore()
        self.setup(**kwargs)

    @property
    def agents(self):
        return AgentIter(self.model, self.positions.keys())

    @property
    def kdtree(self):
        # Create new KDTree if necessary
        if self._cKDTree is None and len(self.agents) > 0:
            self._sorted_agents = []
            self._sorted_agent_points = []
            for a in self.agents:
                self._sorted_agents.append(a)
                self._sorted_agent_points.append(self.positions[a])
            if self._torus:
                self._cKDTree = spatial.cKDTree(self._sorted_agent_points,
                                                boxsize=self.shape)
            else:
                self._cKDTree = spatial.cKDTree(self._sorted_agent_points)
        return self._cKDTree  # Return existing or new KDTree

    # Add and remove agents ------------------------------------------------- #

    def add_agents(self, agents, positions=None, random=False):
        """ Adds agents to the space environment.

        Arguments:
            agents (Sequence of Agent):
                Instance or iterable of agents to be added.
            positions (Sequence of positions, optional):
                The positions of the agents.
                Must have the same length as 'agents',
                with each entry being a position (array of float).
                If none is passed, all positions will be either be zero
                or random based on the argument 'random'.
            random (bool, optional):
                Whether to choose random positions (default False).
        """

        self._cKDTree = None  # Reset KDTree
        if not positions:
            n_agents = len(agents)
            if random:
                positions = [[self.model.random.random() * d_max
                              for d_max in self.shape]
                             for _ in range(n_agents)]
            else:
                positions = [np.zeros(self.ndim) for _ in range(n_agents)]

        for agent, pos in zip(agents, positions):

            pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
            self.positions[agent] = pos  # Add pos to agent_dict

    def remove_agents(self, agents):
        """ Removes agents from the space. """
        self._cKDTree = None  # Reset KDTree
        for agent in make_list(agents):
            del self.positions[agent]  # Remove agent from env

    # Move and select agents ------------------------------------------------ #

    @staticmethod
    def _border_behavior(position, shape, torus):
        # Border behavior

        # Connected - Jump to other side
        if torus:
            for i in range(len(position)):
                while position[i] > shape[i]:
                    position[i] -= shape[i]
                while position[i] < 0:
                    position[i] += shape[i]

        # Not connected - Stop at border
        else:
            for i in range(len(position)):
                if position[i] > shape[i]:
                    position[i] = shape[i]
                elif position[i] < 0:
                    position[i] = 0

    def move_to(self, agent, pos):
        """ Moves agent to new position.

        Arguments:
            agent (Agent): Instance of the agent.
            pos (array_like): New position of the agent.
        """

        self._cKDTree = None  # Reset KDTree
        self._border_behavior(pos, self.shape, self._torus)
        self.positions[agent][...] = pos  # In-place

    def move_by(self, agent, path):
        """ Moves agent to new position, relative to current position.

        Arguments:
            agent (Agent): Instance of the agent.
            path (array_like): Relative change of position.
        """
        pos = [p + c for p, c in zip(self.positions[agent], path)]
        self.move_to(agent, pos)

    def neighbors(self, agent, distance):
        """ Select agent neighbors within a given distance.
        Takes into account wether space is toroidal.

        Arguments:
            agent (Agent): Instance of the agent.
            distance (float):
                Radius around the agent in which to search for neighbors.

        Returns:
            AgentIter: Iterator over the selected neighbors.
        """

        list_ids = self.kdtree.query_ball_point(
            self.positions[agent], distance)

        agents = [self._sorted_agents[list_id] for list_id in list_ids]
        agents.remove(agent)  # Remove original agent
        return AgentIter(self.model, agents)

    def select(self, center, radius):
        """ Select agents within a given area.

        Arguments:
            center (array_like): Coordinates of the center of the search area.
            radius (float): Radius around the center in which to search.

        Returns:
            AgentIter: Iterator over the selected agents.
        """
        if self.kdtree:
            list_ids = self.kdtree.query_ball_point(center, radius)
            agents = [self._sorted_agents[list_id] for list_id in list_ids]
            return AgentIter(self.model, agents)
        else:
            return AgentIter(self.model)
