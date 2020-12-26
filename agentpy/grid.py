"""
Agentpy Grid Module
Content: Classes for spatial grids
"""


import itertools
import numpy as np
import random as rd
from .objects import ApEnv, Agent
from .tools import make_list, make_matrix
from .lists import AgentList


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

    def __init__(self, model, shape, **kwargs):

        super().__init__(model)

        # Convert single-number shape to tuple
        if isinstance(shape, (int, float)):
            shape = (shape, shape)

        self._topology = 'grid'
        self._grid = make_matrix(make_list(shape), AgentList)
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

    def add_agents(self, agents=1, agent_class=Agent, positions=None,  # noqa
                   random=False, **kwargs):
        """ Adds agents to the grid environment.
        See :func:`Environment.add_agents` for standard arguments.
        Additional arguments are listed below.

        Arguments:
            positions(list of tuples, optional): The positions of the added
                agents. List must have the same length as number of agents
                to be added, and each entry must be a tuple with coordinates.
                If none is passed, agents will fill up the grid systematically.
            random(bool, optional):
                If no positions are passed, agents will be placed in random
                locations instead of systematic filling (default False).
        """

        # Standard adding
        new_agents = super().add_agents(agents, agent_class, **kwargs)

        # Extra grid features
        if not positions:
            n_agents = len(new_agents)
            all_positions = list(self.positions())
            n_all_positions = len(list(self.positions()))
            positions = []
            while n_agents > n_all_positions - len(positions):
                positions.extend(all_positions)
            if n_all_positions - len(positions) > 0:
                if random:
                    positions.extend(rd.sample(list(self.positions()), agents))
                else:
                    positions.extend(all_positions[:n_agents])
                # for agent in new_agents:
                #     self._positions[agent] = [0] * self.dim

        for agent, pos in zip(new_agents, positions):
            self._agent_dict[agent] = pos  # Add position to agent_dict
            self._get_pos(pos).append(agent)  # Add agent to position

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

    def _get_attr(self, agent_list, attr_key, mode, fill_empty):
        if len(agent_list) == 0:
            return fill_empty
        if mode == 'single':
            return getattr(agent_list[0], attr_key)
        if mode == 'list':
            return list(getattr(agent_list, attr_key))
        if mode == 'sum':
            return sum(getattr(agent_list, attr_key))

    def attribute(self, attr_key, mode='single', empty=np.nan):
        """ Returns a grid with the value of the attributes of the agents
        in each position.

        Arguments:
            attr_key(str): Name of the attribute.
            mode(str, optional): What to return in for positions with agents:
                If 'single' (default), the attribute of the
                first agent; if 'list', a list of all agents' attributes;
                if 'sum', the sum of all agents' attributes.
            empty(optional): What to return for empty positions
                without agents (default numpy.nan).
        """
        return self.apply(self._get_attr, attr_key, mode, empty)

    def position(self, agent):
        """ Returns position of a passed agent.

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

    def _get_pos_rec(self, grid, pos):
        """ Recursive function to get position. """
        if len(pos) == 1:
            return grid[pos[0]]
        return self._get_pos_rec(grid[pos[0]], pos[1:])

    def _get_pos(self, pos):
        return self._get_pos_rec(self._grid, pos)

    def _get_agents_from_area(self, area, grid):
        subgrid = grid[area[0][0]:area[0][1] + 1]
        # Detect last row (must have AgentLists)
        if isinstance(subgrid[0], AgentList):
            # Flatten list of AgentLists to list of agents
            return [y for x in subgrid for y in x]
        objects = []
        for row in subgrid:
            objects.extend(self._get_agents_from_area(area[1:], row))
        return objects

    def get_agents(self, area):
        """ Returns an :class:`AgentList` with agents
        in the selected positions or area.
        To select all agents, use :attr:`Grid.agents`.

        Arguments:
            area(list of integers or tuples):
                Area from which agents should be gathered.
                Can either indicate a single position `[x, y, ...]`
                or an area `[(x_start, x_end), (y_start, y_end), ...]`.
        """
        if isinstance(area[0], int):
            return Agentlist(self._get_pos(pos))  # Soft copy
        else:
            return AgentList(self._get_agents_from_area(area, self._grid))

    def move_agent(self, agent, position):
        """ Moves agent to new position.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
            position(list of int): New position of the agent.
        """
        if isinstance(agent, int):
            agent = self.model.get_obj(agent)
        self._get_pos(self._agent_dict[agent]).remove(agent)
        self._get_pos(position).append(agent)
        self._agent_dict[agent] = position  # Document new position

    def items(self, area=None):
        """ Returns iterator with tuples of style: (position, agents). """
        # TODO Test efficiency, alternative mode for area=None would be faster
        p_it_1, p_it_2 = itertools.tee(self.positions(area))  # Copy iterator
        return zip(p_it_1, [self._get_pos(pos) for pos in p_it_2])

    def _get_neighbors4(self, pos, grid, dist=1):
        """ Return agents in diamond-shaped area around pos."""
        subgrid = grid[max(0, pos[0] - dist):pos[0] + dist + 1]
        # TODO Includes agents in the middle!
        if len(pos) == 1:
            return [y for x in subgrid for y in x]  # flatten list
        objects = []
        for row, dist in zip(subgrid, [0, 1, 0]):
            objects.extend(self._get_neighbors4(pos[1:], row, dist))
        return objects

    def _get_neighbors8(self, pos, grid):
        """ Return agents in square-shaped area around pos."""
        subgrid = grid[max(0, pos[0] - 1):pos[0] + 2]
        # TODO Includes agents in the middle!
        if len(pos) == 1:
            return [y for x in subgrid for y in x]
        objects = []
        for row in subgrid:
            objects.extend(self._get_neighbors8(pos[1:], row))
        return objects

    def neighbors(self, agent, mode='cube'):
        """ Returns agent neighbors.

        Arguments:
            agent(int or Agent): Id or instance of the agent.
            mode(str, optional): Selection mode (default 'cube').
                Diagonal neighbors are included if mode is 'cube',
                or excluded if mode is 'diamond'.
        """
        if isinstance(agent, int):
            agent = self.model.get_obj(agent)
        if mode is None or mode == 'cube':  # Include diagonal neighbors
            agents = self._get_neighbors8(self._agent_dict[agent], self._grid)
        elif mode == 'diamond':  # Do not include diagonal neighbors
            agents = self._get_neighbors4(self._agent_dict[agent], self._grid)
        else:
            raise ValueError(f"Grid.neighbors has no mode '{mode}'.")
        agents.remove(agent)  # Remove the original agent TODO Automatic
        return agents
