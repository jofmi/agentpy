"""
Agentpy Grid Module
Content: Class for discrete spatial environments
"""

# There is much room for optimization in this module. Contributions welcome! :)
# TODO __init__: Add argument connect_borders
# TODO _get_diamond: distance argument, exclude original agent, & limit dmax
# TODO Method for distance between agents
# TODO items: reverse argument

import itertools
import numpy as np
import random as rd
import collections.abc as abc
from .objects import ApEnv, Agent
from .tools import make_list, make_matrix
from .lists import AgentList


class Grid(ApEnv):
    """ Environment that contains agents with a discrete spatial topology.
    Every location consists of an :class:`AgentList` that can hold
    zero, one, or more agents.
    To add new grid environments to a model, use :func:`Model.add_grid`.
    For a continuous spatial topology, use :class:`Space`.

    This class can be used as a parent class for custom grid types.
    All agentpy model objects call the method :func:`setup` after creation,
    and can access class attributes like dictionary items.
    See :class:`Environment` for general properties of all environments.

    Arguments:
        model (Model): The model instance.
        shape (tuple of int): Size of the grid.
            The length of the tuple defines the number of dimensions,
            and the values in the tuple define the length of each dimension.
        **kwargs: Will be forwarded to :func:`Grid.setup`.

    Attributes:
        grid (list of lists): Matrix of :class:`AgentList`.
        shape (tuple of int): Length of each grid dimension.
        dim (int): Number of dimensions.
    """

    def __init__(self, model, shape, **kwargs):

        super().__init__(model)

        self._topology = 'grid'
        self._grid = make_matrix(shape, AgentList)
        self._agent_dict = {}
        self._shape = shape
        self._set_var_ignore()
        self.setup(**kwargs)

    @property
    def grid(self):
        return self._grid

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return len(self._shape)

    def add_agents(self, agents=1, agent_class=Agent, positions=None,  # noqa
                   random=False, generator=None, **kwargs):
        """ Adds agents to the grid environment, and returns new agents.
        See :func:`Environment.add_agents` for standard arguments.
        Additional arguments are listed below.

        Arguments:
            positions(list of tuple, optional): The positions of the added
                agents. List must have the same length as number of agents
                to be added, and each entry must be a tuple with coordinates.
                If none is passed, agents will fill up the grid systematically.
            random(bool, optional):
                If no positions are passed, agents will be placed in random
                locations instead of systematic filling (default False).
            generator (numpy.random.Generator, optional):
                Random number generator that is used if 'random' is True.
                If none is passed, :obj:`Model.random` is used.
        """

        # Standard adding
        new_agents = super().add_agents(agents, agent_class, **kwargs)

        # Extra grid features
        if not positions:
            n_agents = len(new_agents)
            available_positions = list(self.positions())

            # Extend positions if necessary
            while n_agents > len(available_positions):
                available_positions.extend(available_positions)

            # Fill agent positions
            positions = []
            if random:
                generator = generator if generator else self.model.random
                positions.extend(generator.choice(available_positions,
                                                  n_agents, replace=False))
            else:
                positions.extend(available_positions[:n_agents])

        for agent, pos in zip(new_agents, positions):
            pos = tuple(pos)  # Ensure that position is tuple
            self._agent_dict[agent] = pos  # Add position to agent_dict
            self._get_agents_from_pos(pos).append(agent)  # Add agent to pos

        return new_agents

    def remove_agents(self, agents):
        """ Removes agents from the environment. """
        for agent in make_list(agents):
            self._get_agents_from_pos(self.position(agent)).remove(agent)
            del self._agent_dict[agent]
        super().remove_agents(agents)

    @staticmethod
    def _apply(grid, func, *args, **kwargs):
        if not isinstance(grid[0], AgentList):
            return [Grid._apply(subgrid, func, *args, **kwargs)
                    for subgrid in grid]
        else:
            return [func(i, *args, **kwargs) for i in grid]

    def apply(self, func, *args, **kwargs):
        """ Applies a function to all grid positions,
        and returns grid with return values. """
        return self._apply(self.grid, func, *args, **kwargs)

    @staticmethod
    def _get_attr(agent_list, attr_key, sum_values, fill_empty):
        if len(agent_list) == 0:
            return fill_empty
        if sum_values:
            return sum(getattr(agent_list, attr_key))
        else:
            return list(getattr(agent_list, attr_key))

    def attribute(self, attr_key, sum_values=True, empty=np.nan):
        """ Returns a grid with the value of the attributes of the agents
        in each position.

        Arguments:
            attr_key(str): Name of the attribute.
            sum_values(str, optional): What to return in a position where there
                are multiple agents. If True (default), the sum of attributes.
                If False, a list of attributes.
            empty(optional): What to return for empty positions
                without agents (default numpy.nan).
        """
        return self.apply(self._get_attr, attr_key, sum_values, empty)

    def position(self, agent):
        """ Returns tuple with position of a passed agent.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
        """
        if isinstance(agent, int):
            agent = self.model.get_obj(agent)
        return self._agent_dict[agent]

    def positions(self, area=None):
        """Returns iterable of all grid positions in area.

        Arguments:
            area(list of tuples, optional):
                Area of positions that should be returned.
                If none is passed, the whole grid is selected.
                Style: `[(x_start, x_end), (y_start, y_end), ...]`
        """
        if area is None:
            return itertools.product(*[range(x) for x in self._shape])
        else:
            return itertools.product(*[range(x, y+1) for x, y in area])

    def _get_agents_from_pos_rec(self, grid, pos):
        """ Recursive function to get position. """
        if len(pos) == 1:
            return grid[pos[0]]
        return self._get_agents_from_pos_rec(grid[pos[0]], pos[1:])

    def _get_agents_from_pos(self, pos):
        """ Return content of a position. """
        return self._get_agents_from_pos_rec(self._grid, pos)

    def _get_agents_from_area(self, area, grid):
        """ Recursive function to get agents in area. """
        subgrid = grid[area[0][0]:area[0][1] + 1]
        # Detect last row (must have AgentLists)
        if isinstance(subgrid[0], AgentList):
            # Flatten list of AgentLists to list of agents
            return [y for x in subgrid for y in x]
        objects = []
        for row in subgrid:
            objects.extend(self._get_agents_from_area(area[1:], row))
        return objects

    def get_agents(self, area=None):
        """ Returns an :class:`AgentList` with agents
        in the selected positions or area.

        Arguments:
            area(tuple of integers or tuples):
                Area from which agents should be gathered.
                Can either indicate a single position `[x, y, ...]`
                or an area `[(x_start, x_end), (y_start, y_end), ...]`.
        """
        if area is None:
            return self.agents
        if not isinstance(area, abc.Iterable):
            raise ValueError(f"area '{area}' is not iterable.")
        if isinstance(area[0], int):  # Soft copy
            return AgentList(self._get_agents_from_pos(area), model=self.model)
        else:
            return AgentList(
                self._get_agents_from_area(area, self._grid), model=self.model)

    def move_agent(self, agent, position):
        """ Moves agent to new position.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
            position(list of int): New position of the agent.
        """
        if isinstance(agent, int):
            agent = self.model.get_obj(agent)
        self._get_agents_from_pos(self._agent_dict[agent]).remove(agent)
        self._get_agents_from_pos(position).append(agent)
        self._agent_dict[agent] = position  # Document new position

    def items(self, area=None):
        """ Returns iterator with tuples of style: (position, agents). """
        p_it_1, p_it_2 = itertools.tee(self.positions(area))  # Copy iterator
        return zip(p_it_1, [self._get_agents_from_pos(pos)
                            for pos in p_it_2])

    def _get_diamond(self, pos, grid, dist=1):
        """ Return agents in diamond-shaped area around pos."""
        subgrid = grid[max(0, pos[0] - dist):pos[0] + dist + 1]
        if len(pos) == 1:
            return [y for x in subgrid for y in x]  # flatten list
        objects = []
        for row, dist in zip(subgrid, [0, 1, 0]):
            objects.extend(self._get_diamond(pos[1:], row, dist))
        return objects

    def neighbors(self, agent, distance=1, diagonal=True):
        """ Returns agent neighbors.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
            distance(int, optional): Number of positions to cover in each
                direction.
            diagonal(bool, optional):
                If True (default), diagonal neighbors are included.
                If False, only direct neighbors are included
                (currently only works with distance == 1).
        """
        if isinstance(agent, int):
            agent = self.model.get_obj(agent)
        if diagonal:  # Include diagonal neighbors
            a_pos = self._agent_dict[agent]
            area = [(max(p-distance, 0),
                     min(p+distance, dmax))
                    for p, dmax in zip(a_pos, self._shape)]
            agents = self.get_agents(area)
        else:  # Do not include diagonal neighbors
            agents = self._get_diamond(self._agent_dict[agent], self._grid)
        agents.remove(agent)  # Remove original agent
        return AgentList(agents, model=self.model)
