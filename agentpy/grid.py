"""
Agentpy Grid Module
Content: Class for discrete spatial environments
"""

import itertools
import numpy as np
import random as rd
import collections.abc as abc
import numpy.lib.recfunctions as rfs
from .objects import Object
from .tools import make_list, make_matrix, AgentpyError, ListDict
from .sequences import AgentSet, AgentIter, AgentList


class _IterArea:
    """ Iteratable object that takes either a numpy matrix or an iterable
    as an input. If the object is an ndarray, it is flattened and iterated
    over the contents of each element chained together. Otherwise, it is
    simply iterated over the object.

    Arguments:
        area: Area of sets of elements.
        exclude: Element to exclude. Assumes that element is in area.
    """

    def __init__(self, area, exclude=None):
        self.area = area
        self.exclude = exclude

    def __len__(self):
        if isinstance(self.area, np.ndarray):
            len_ = sum([len(s) for s in self.area.flat])
        else:
            len_ = len(self.area)
        if self.exclude:
            len_ -= 1  # Assumes that exclude is in Area
        return len_

    def __iter__(self):
        if self.exclude:
            if isinstance(self.area, np.ndarray):
                return itertools.filterfalse(
                    lambda x: x is self.exclude,
                    itertools.chain.from_iterable(self.area.flat)
                )
            else:
                return itertools.filterfalse(
                    lambda x: x is self.exclude, self.area)
        else:
            if isinstance(self.area, np.ndarray):
                return itertools.chain.from_iterable(self.area.flat)
            else:
                return iter(self.area)


class GridIter(AgentIter):
    """ Iterator over objects in :class:`Grid` that supports slicing.

    Examples:

         Create a model with a 10 by 10 grid
         with one agent in each position::

            model = ap.Model()
            agents = ap.AgentList(model, 100)
            grid = ap.Grid(model, (10, 10))
            grid.add_agents(agents)

        The following returns an iterator over the agents in all position::

            >>> grid.agents
            GridIter (100 objects)

        The following returns an iterator over the agents
        in the top-left quarter of the grid::

            >>> grid.agents[0:5, 0:5]
            GridIter (25 objects)
    """

    def __init__(self, model, iter_, items):
        super().__init__(model, iter_)
        object.__setattr__(self, '_items', items)

    def __getitem__(self, item):
        sub_area = self._items[item]
        return GridIter(self._model, _IterArea(sub_area), sub_area)


class Grid(Object):
    """ Environment that contains agents with a discrete spatial topology,
    supporting multiple agents and attribute fields per cell.
    For a continuous spatial topology, see :class:`Space`.

    This class can be used as a parent class for custom grid types.
    All agentpy model objects call the method :func:`setup` after creation,
    and can access class attributes like dictionary items.

    Arguments:
        model (Model):
            The model instance.
        shape (tuple of int):
            Size of the grid.
            The length of the tuple defines the number of dimensions,
            and the values in the tuple define the length of each dimension.
        torus (bool, optional):
            Whether to connect borders (default False).
            If True, the grid will be toroidal, meaning that agents who
            move over a border will re-appear on the opposite side.
            If False, they will remain at the edge of the border.
        track_empty (bool, optional):
            Whether to keep track of empty cells (default False).
            If true, empty cells can be accessed via :obj:`Grid.empty`.
        check_border (bool, optional):
            Ensure that agents stay within border (default True).
            Can be set to False for faster performance.
        **kwargs: Will be forwarded to :func:`Grid.setup`.

    Attributes:
        agents (GridIter):
            Iterator over all agents in the grid.
        positions (dict of Agent):
            Dictionary linking each agent instance to its position.
        grid (numpy.rec.array):
            Structured numpy record array with a field 'agents'
            that holds an :class:`AgentSet` in each position.
        shape (tuple of int):
            Length of each dimension.
        ndim (int):
            Number of dimensions.
        all (list):
            List of all positions in the grid.
        empty (ListDict):
            List of unoccupied positions, only available
            if the Grid was initiated with `track_empty=True`.
    """

    @staticmethod
    def _agent_field(field_name, shape, model):
        # Prepare structured array filled with empty agent sets
        array = np.empty(shape, dtype=[(field_name, object)])
        it = np.nditer(array, flags=['refs_ok', 'multi_index'])
        for _ in it:
            array[it.multi_index] = AgentSet(model)
        return array

    def __init__(self, model, shape, torus=False,
                 track_empty=False, check_border=True, **kwargs):

        super().__init__(model)

        self._track_empty = track_empty
        self._check_border = check_border
        self._torus = torus

        self.positions = {}
        self.grid = np.rec.array(self._agent_field('agents', shape, model))
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.all = list(itertools.product(*[range(x) for x in shape]))
        self.empty = ListDict(self.all) if track_empty else None

        self._set_var_ignore()
        self.setup(**kwargs)

    @property
    def agents(self):
        return GridIter(self.model, self.positions.keys(), self.grid.agents)

    # Add and remove agents ------------------------------------------------- #

    def _add_agent(self, agent, position, field):
        position = tuple(position)
        self.grid[field][position].add(agent)  # Add agent to grid
        self.positions[agent] = position  # Add agent position to dict

    def add_agents(self, agents, positions=None, random=False, empty=False):
        """ Adds agents to the grid environment.

        Arguments:
            agents (Sequence of Agent):
                Iterable of agents to be added.
            positions (Sequence of positions, optional):
                The positions of the agents.
                Must have the same length as 'agents',
                with each entry being a tuple of integers.
                If none is passed, positions will be chosen automatically
                based on the arguments 'random' and 'empty':

                - random and empty:
                  Random selection without repetition from `Grid.empty`.
                - random and not empty:
                  Random selection with repetition from `Grid.all`.
                - not random and empty:
                  Iterative selection from `Grid.empty`.
                - not random and not empty:
                  Iterative selection from `Grid.all`.

            random (bool, optional):
                Whether to choose random positions (default False).
            empty (bool, optional):
                Whether to choose only empty cells (default False).
                Can only be True if Grid was initiated with `track_empty=True`.
        """

        field = 'agents'

        if empty and not self._track_empty:
            raise AgentpyError(
                "To use 'Grid.add_agents()' with 'empty=True', "
                "Grid must be iniated with 'track_empty=True'.")

        # Choose positions
        if positions:
            pass
        elif random:
            n = len(agents)
            if empty:
                positions = self.model.random.sample(self.empty, k=n)
            else:
                positions = self.model.random.choices(self.all, k=n)
        else:
            if empty:
                positions = list(self.empty)  # Soft copy
            else:
                positions = itertools.cycle(self.all)

        if empty and len(positions) < len(agents):
            raise AgentpyError("Cannot add more agents than empty positions.")

        if self._track_empty:
            for agent, position in zip(agents, positions):
                self._add_agent(agent, position, field)
                if position in self.empty:
                    self.empty.remove(position)
        else:
            for agent, position in zip(agents, positions):
                self._add_agent(agent, position, field)

    def remove_agents(self, agents):
        """ Removes agents from the environment. """
        for agent in make_list(agents):
            pos = self.positions[agent]  # Get position
            self.grid.agents[pos].remove(agent)  # Remove agent from grid
            del self.positions[agent]  # Remove agent from position dict
            if self._track_empty:
                self.empty.append(pos)  # Add position to free spots

    # Move and select agents ------------------------------------------------ #

    @staticmethod
    def _border_behavior(position, shape, torus):
        position = list(position)
        # Connected - Jump to other side
        if torus:
            for i in range(len(position)):
                while position[i] > shape[i]-1:
                    position[i] -= shape[i]
                while position[i] < 0:
                    position[i] += shape[i]

        # Not connected - Stop at border
        else:
            for i in range(len(position)):
                if position[i] > shape[i]-1:
                    position[i] = shape[i]-1
                elif position[i] < 0:
                    position[i] = 0
        return tuple(position)

    def move_to(self, agent, pos):
        """ Moves agent to new position.

        Arguments:
            agent (Agent): Instance of the agent.
            pos (tuple of int): New position of the agent.
        """

        pos_old = self.positions[agent]
        if pos != pos_old:

            # Grid options
            if self._check_border:
                pos = self._border_behavior(pos, self.shape, self._torus)
            if self._track_empty:
                if len(self.grid.agents[pos_old]) == 1:
                    if pos in self.empty:
                        self.empty.replace(pos, pos_old)
                    else:
                        self.empty.append(pos_old)
                elif pos in self.empty:
                    self.empty.remove(pos)

            self.grid.agents[pos_old].remove(agent)
            self.grid.agents[pos].add(agent)
            self.positions[agent] = pos

    def move_by(self, agent, path):
        """ Moves agent to new position, relative to current position.

        Arguments:
            agent (Agent): Instance of the agent.
            path (tuple of int): Relative change of position.
        """
        pos = [p + c for p, c in zip(self.positions[agent], path)]
        self.move_to(agent, tuple(pos))

    def neighbors(self, agent, distance=1):
        """ Select neighbors of an agent within a given distance.

        Arguments:
            agent (Agent): Instance of the agent.
            distance (int, optional):
                Number of cells to cover in each direction,
                including diagonally connected cells (default 1).

        Returns:
            AgentIter: Iterator over the selected neighbors.
        """

        pos = self.positions[agent]

        # TODO Change method upon initiation
        # Case 1: Toroidal
        if self._torus:
            slices = [(p-distance, p+distance+1) for p in pos]
            new_slices = []
            for (x_from, x_to), x_max in zip(slices, self.shape):
                if x_to > x_max and x_from < 0:
                    sl_tupl = [(0, x_max)]
                elif x_to > x_max:
                    if x_to - x_max >= x_from:
                        sl_tupl = [(0, x_max)]
                    else:
                        sl_tupl = [(x_from, x_max), (0, x_to - x_max)]
                elif x_from < 0:
                    if x_max + x_from <= x_to:
                        sl_tupl = [(0, x_max)]
                    else:
                        sl_tupl = [(x_max + x_from, x_max), (0, x_to)]
                else:
                    sl_tupl = [(x_from, x_to)]
                new_slices.append(sl_tupl)
            list_of_slices = list(itertools.product(*new_slices))
            areas = []
            for slices in list_of_slices:
                slices = tuple([slice(*sl) for sl in slices])
                areas.append(self.grid.agents[slices])
            # TODO Exclude in every area inefficient
            area_iters = [_IterArea(area, exclude=agent) for area in areas]
            # TODO Can only be iterated on once
            return AgentIter(self.model,
                             itertools.chain.from_iterable(area_iters))

        # Case 2: Non-toroidal
        else:
            slices = tuple([slice(p-distance if p-distance >= 0 else 0,
                                  p+distance+1) for p in pos])
            area = self.grid.agents[slices]
            # Iterator over all agents in area, exclude original agent
            return AgentIter(self.model, _IterArea(area, exclude=agent))

    # Fields and attributes ------------------------------------------------- #

    def apply(self, func, field='agents'):
        """ Applies a function to each grid position,
        end returns an `numpy.ndarray` of return values.

        Arguments:
            func (function): Function that takes cell content as input.
            field (str, optional): Field to use (default 'agents').
        """
        return np.vectorize(func)(self.grid[field])

    def attr_grid(self, attr_key, otypes='f', field='agents'):
        """ Returns a grid with the value of the attribute of the agent
        in each position, using :class:`numpy.vectorize`.
        Positions with no agent will contain `numpy.nan`.
        Should only be used for grids with zero or one agents per cell.
        Other kinds of attribute grids can be created with :func:`Grid.apply`.

        Arguments:
            attr_key (str): Name of the attribute.
            otypes (str or list of dtypes, optional):
                Data type of returned grid (default float).
                For more information, see :class:`numpy.vectorize`.
            field (str, optional): Field to use (default 'agents').
        """

        f = np.vectorize(
            lambda x: getattr(next(iter(x)), attr_key) if x else np.nan,
            otypes=otypes)
        return f(self.grid[field])

    def add_field(self, key, values=None):
        """
        Add an attribute field to the grid.

        Arguments:
            key (str):
                Name of the field.
            values (optional):
                Single value or :class:`numpy.ndarray`
                of values (default None).
        """

        if not isinstance(values, (np.ndarray, list)):
            values = np.full(sum(self.shape), fill_value=values)
        if len(values.shape) > 1:
            values = values.reshape(-1)

        # Create attribute as a numpy field
        self.grid = rfs.append_fields(
            self.grid, key, values, usemask=False, asrecarray=True
            ).reshape(self.grid.shape)

        # Create attribute as reference to field
        setattr(self, key, self.grid[key])

    def del_field(self, key):
        """
        Delete a attribute field from the grid.

        Arguments:
            key (str): Name of the field.
        """

        self.grid = rfs.drop_fields(
            self.grid, key, usemask=False, asrecarray=True)
        delattr(self, key)
