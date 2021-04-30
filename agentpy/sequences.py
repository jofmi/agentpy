"""
Agentpy Lists Module
Content: Lists for objects, environments, and agents
"""

# TODO Seperate AgentIter & AgentGroupIter

import itertools
import agentpy as ap
import numpy as np
from .tools import AgentpyError
from collections.abc import Sequence as PySequence


class Sequence:
    """ Base class for agenpty sequences. """

    def __repr__(self):
        s = 's' if len(self) != 1 else ''
        return f"{type(self).__name__} ({len(self)} object{s})"

    def __getattr__(self, name):
        # TODO This breaks numpy conversion because of __array_struct__ lookup
        """ Return callable list of attributes """
        if name[0] == '_':  # Allows for numpy conversion
            super().__getattr__(name)  #raise AttributeError('test')
        else:
            return AttrIter(self, attr=name)

    def _set(self, key, value):
        object.__setattr__(self, key, value)

    @staticmethod
    def _obj_gen(model, n, cls, *args, **kwargs):
        if cls is None:
            cls = ap.Agent
        for _ in range(n):
            yield cls(model, *args, **kwargs)


# Attribute List ------------------------------------------------------------ #

class AttrIter(Sequence, PySequence):
    """ Iterator over an attribute of objects in a sequence.
    Length, items access, and representation work like with a normal list.
    Calls are forwarded to each entry and return a list of return values.
    Boolean operators are applied to each entry and return a list of bools.
    Arithmetic operators are applied to each entry and return a new list.
    If applied to another `AttrList`, the first entry of the first list
    will be matched with the first entry of the second list, and so on.
    Else, the same value will be applied to each entry of the list.
    See :class:`AgentList` for examples.
    """

    def __init__(self, source, attr=None):
        self.source = source
        self.attr = attr

    def __repr__(self):
        return repr(list(self))

    @staticmethod
    def _iter_attr(a, s):
        for o in s:
            yield getattr(o, a)

    def __iter__(self):
        """ Iterate through source list based on attribute. """
        if self.attr:
            return self._iter_attr(self.attr, self.source)
        else:
            return iter(self.source)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, key):
        """ Get item from source list. """
        return getattr(self.source[key], self.attr)

    def __setitem__(self, key, value):
        """ Set item to source list. """
        setattr(self.source[key], self.attr, value)

    def __call__(self, *args, **kwargs):
        return [func_obj(*args, **kwargs) for func_obj in self]

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
        if isinstance(v, AttrIter):
            return AttrIter([x + y for x, y in zip(self, v)])
        else:
            return AttrIter([x + v for x in self])

    def __sub__(self, v):
        if isinstance(v, AttrIter):
            return AttrIter([x - y for x, y in zip(self, v)])
        else:
            return AttrIter([x - v for x in self])

    def __mul__(self, v):
        if isinstance(v, AttrIter):
            return AttrIter([x * y for x, y in zip(self, v)])
        else:
            return AttrIter([x * v for x in self])

    def __truediv__(self, v):
        if isinstance(v, AttrIter):
            return AttrIter([x / y for x, y in zip(self, v)])
        else:
            return AttrIter([x / v for x in self])

    def __iadd__(self, v):
        return self + v

    def __isub__(self, v):
        return self - v

    def __imul__(self, v):
        return self * v

    def __itruediv__(self, v):
        return self / v


# Object Containers --------------------------------------------------------- #

class AgentList(Sequence, list):
    """ List of agentpy objects.
    Attribute calls and assignments are applied to all agents
    and return an :class:`AttrIter` with the attributes of each agent.
    This also works for method calls, which returns a list of return values.
    Arithmetic operators can further be used to manipulate agent attributes,
    and boolean operators can be used to filter the list based on agents'
    attributes. Standard :class:`list` methods can also be used.

    Arguments:
        model (Model): The model instance.
        objs (int or Sequence, optional):
            An integer number of new objects to be created,
            or a sequence of existing objects (default empty).
        cls (type, optional): Class for the creation of new objects.
        *args: Forwarded to the constructor of the new objects.
        **kwargs: Forwarded to the constructor of the new objects.

    Examples:

        Prepare an :class:`AgentList` with three agents::

            >>> model = ap.Model()
            >>> agents = model.add_agents(3)
            >>> agents
            AgentList [3 agents]

        The assignment operator can be used to set a variable for each agent.
        When the variable is called, an :class:`AttrList` is returned::

            >>> agents.x = 1
            >>> agents.x
            AttrList of 'x': [1, 1, 1]

        One can also set different variables for each agent
        by passing another :class:`AttrList`::

            >>> agents.y = ap.AttrIter([1, 2, 3])
            >>> agents.y
            AttrList of 'y': [1, 2, 3]

        Arithmetic operators can be used in a similar way.
        If an :class:`AttrList` is passed, different values are used for
        each agent. Otherwise, the same value is used for all agents::

            >>> agents.x = agents.x + agents.y
            >>> agents.x
            AttrList of 'x': [2, 3, 4]

            >>> agents.x *= 2
            >>> agents.x
            AttrList of 'x': [4, 6, 8]

        Attributes of specific agents can be changed through setting items::

            >>> agents.x[2] = 10
            >>> agents.x
            AttrList of 'x': [4, 6, 10]

        Boolean operators can be used to select a subset of agents::

            >>> subset = agents(agents.x > 5)
            >>> subset
            AgentList [2 agents]

            >>> subset.x
            AttrList of attribute 'x': [6, 8]
    """

    def __init__(self, model, objs=(), cls=None, *args, **kwargs):
        if isinstance(objs, int):
            objs = self._obj_gen(model, objs, cls, *args, **kwargs)
        super().__init__(objs)
        super().__setattr__('model', model)
        super().__setattr__('ndim', 1)

    def __setattr__(self, name, value):
        if isinstance(value, AttrIter):
            # Apply each value to each agent
            for obj, v in zip(self, value):
                setattr(obj, name, v)
        else:
            # Apply single value to all agents
            for obj in self:
                setattr(obj, name, value)

    def select(self, selection):
        """ Returns a new :class:`AgentList` based on `selection`.

        Arguments:
            selection (list of bool): List with same length as the agent list.
                Positions that return True will be selected.
        """
        return AgentList(self.model, [a for a, s in zip(self, selection) if s])

    # TODO offer both choices
    def random(self, n=1, replace=False, weights=None, shuffle=True):
        """ Creates a random sample of agents,
        using :func:`numpy.random.Generator.choice`.
        Argument descriptions are adapted from :obj:`numpy.random`.
        Returns a new :class:`AgentList` with the selected agents.

        Arguments:
            n (int, optional): Number of agents (default 1).
            replace (bool, optional):
                Whether the sample is with or without replacement.
                Default is False, meaning that every agent can
                only be selected once.
            weights (1-D array_like, optional):
                The probabilities associated with each agent.
                If not given the sample assumes a uniform distribution
                over all agents.
            shuffle (bool, optional):
                Whether the sample is shuffled
                when sampling without replacement.
                Default is True, False provides a speedup.
        """

        # Choice is not applied to list directly because it would convert it to
        # a numpy array, which takes much more time than the current solution.
        indices = self.model.nprandom.choice(
            len(self), size=n, replace=replace, p=weights, shuffle=shuffle)
        selection = AgentList(self.model, [self[i] for i in indices])
        return selection

    def sort(self, var_key, reverse=False):
        """ Sorts the list in-place, and returns self.

        Arguments:
            var_key (str): Attribute of the lists' objects, based on which
                the list will be sorted from lowest value to highest.
            reverse (bool, optional): Reverse sorting (default False).
        """
        super().sort(key=lambda x: x[var_key], reverse=reverse)
        return self

    def shuffle(self):
        """ Shuffles the list in-place, and returns self. """
        self.model.random.shuffle(self)
        return self


class AgentGroup(Sequence, PySequence):
    """ Ordered collection of agentpy objects.
    This class behaves similar to the :class:`AgentList` in most aspects,
    but provide extra features and better performance for object removal.

    The following aspects are different:

    - Faster removal of objects.
    - Faster lookup if object is part of group.
    - No duplicates are allowed.
    - The order of agents in the group cannot be changed.
    - Removal of agents changes the order of the group.
    - The method :class:`AgentGroup.buffer` makes it possible to
      remove objects from the group while iterating over the group.
    - The method :class:`AgentGroup.shuffle` returns an iterator
      instead of shuffling in-place.

    Arguments:
        model (Model): The model instance.
        objs (int or Sequence, optional):
            An integer number of new objects to be created,
            or a sequence of existing objects (default empty).
        cls (type, optional): Class for the creation of new objects.
        *args: Forwarded to the constructor of the new objects.
        **kwargs: Forwarded to the constructor of the new objects.

    """

    # TODO Random, select, setattr

    def __init__(self, model, objs=(), cls=None, *args, **kwargs):
        if isinstance(objs, int):
            objs = self._obj_gen(model, objs, cls, *args, **kwargs)

        self._set('model', model)
        self._set('ndim', 1)
        self._set('items', [])
        self._set('item_to_position', {})

        self.model = model
        self.item_to_position = {}
        self.items = []
        for obj in objs:
            self.append(obj)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def __contains__(self, item):
        return item in self.item_to_position

    def __setattr__(self, name, value):
        if isinstance(value, AttrIter):
            # Apply each value to each agent
            for obj, v in zip(self.items, value):
                setattr(obj, name, v)
        else:
            # Apply single value to all agents
            for obj in self.items:
                setattr(obj, name, value)

    def append(self, item):
        """ Add an object to the group. """
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def replace(self, old_item, new_item):
        """ Replace an object with another. """
        position = self.item_to_position.pop(old_item)
        self.item_to_position[new_item] = position
        self.items[position] = new_item

    def pop(self, index):
        """ Remove an object from the group by index. """
        self.remove(self[index])

    def remove(self, item):
        """ Remove an object from the group by instance. """
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def shuffle(self):
        """ Return :class:`AgentIter` over the content of the group
         with the order of objects being shuffled. """
        return AgentGroupIter(self.model, self, shuffle=True)

    def buffer(self):
        """ Return :class:`AgentIter` over the content of the group
         that supports deletion of objects from the group during iteration. """
        return AgentGroupIter(self.model, self, buffer=True)


class AgentSet(Sequence, set):
    """ Unordered collection of agentpy objects.

    Arguments:
        model (Model): The model instance.
        objs (int or Sequence, optional):
            An integer number of new objects to be created,
            or a sequence of existing objects (default empty).
        cls (type, optional): Class for the creation of new objects.
        *args: Forwarded to the constructor of the new objects.
        **kwargs: Forwarded to the constructor of the new objects.
    """

    def __init__(self, model, objs=(), cls=None, *args, **kwargs):
        if isinstance(objs, int):
            objs = self._obj_gen(model, objs, cls, *args, **kwargs)
        super().__init__(objs)
        super().__setattr__('model', model)
        super().__setattr__('ndim', 1)


class AgentIter(Sequence):
    """ Iterator over agentpy objects. """

    def __init__(self, source=()):
        object.__setattr__(self, '_source', source)

    def __iter__(self):
        return iter(self._source)

    def __len__(self):
        return len(self._source)

    def __setattr__(self, name, value):
        if isinstance(value, AttrIter):
            # Apply each value to each agent
            for obj, v in zip(self, value):
                setattr(obj, name, v)
        else:
            # Apply single value to all agents
            for obj in self:
                setattr(obj, name, value)


class AgentGroupIter(AgentIter):
    """ Iterator over agentpy objects in an :class:`AgentGroup`. """

    def __init__(self, model, source=(), shuffle=False, buffer=False):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_source', source)
        object.__setattr__(self, '_shuffle', shuffle)
        object.__setattr__(self, '_buffer', buffer)

    def __iter__(self):
        if self._buffer:
            return self._buffered_iter()
        elif self._shuffle:
            items = self._source.items.copy()
            self._model.random.shuffle(items)
            return iter(items)
        else:
            return iter(self._source)

    def buffer(self):
        object.__setattr__(self, '_buffer', True)

    def shuffle(self):
        object.__setattr__(self, '_shuffle', True)

    def _buffered_iter(self):
        """ Iterate over source. """
        items = self._source.items.copy()
        if self._shuffle:
            self._model.random.shuffle(items)
        for a in items:
            if a in self._source:
                yield a