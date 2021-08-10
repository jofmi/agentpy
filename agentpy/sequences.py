"""
Agentpy Lists Module
Content: Lists for objects, environments, and agents
"""

import itertools
import agentpy as ap
import numpy as np
from .tools import AgentpyError, ListDict
from collections.abc import Sequence


class AgentSequence:
    """ Base class for agenpty sequences. """

    def __repr__(self):
        len_ = len(list(self))
        s = 's' if len_ != 1 else ''
        return f"{type(self).__name__} ({len_} object{s})"

    def __getattr__(self, name):
        """ Return callable list of attributes """
        if name[0] == '_':  # Private variables are looked up normally
            # Gives numpy conversion correct error for __array_struct__ lookup
            super().__getattr__(name)
        else:
            return AttrIter(self, attr=name)

    def _set(self, key, value):
        object.__setattr__(self, key, value)

    @staticmethod
    def _obj_gen(model, n, cls, *args, **kwargs):
        """ Generate objects for sequence. """

        if cls is None:
            cls = ap.Agent

        if args != tuple():
            raise AgentpyError(
                "Sequences no longer accept extra arguments without a keyword."
                f" Please assign a keyword to the following arguments: {args}")

        for i in range(n):
            # AttrIter values get broadcasted among agents
            i_kwargs = {k: arg[i] if isinstance(arg, AttrIter) else arg
                        for k, arg in kwargs.items()}
            yield cls(model, **i_kwargs)


# Attribute List ------------------------------------------------------------ #

class AttrIter(AgentSequence, Sequence):
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
        if self.attr:
            return getattr(self.source[key], self.attr)
        else:
            return self.source[key]

    def __setitem__(self, key, value):
        """ Set item to source list. """
        if self.attr:
            setattr(self.source[key], self.attr, value)
        else:
            self.source[key] = value

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

def _random(model, gen, obj_list, n=1, replace=False):
    """ Creates a random sample of agents.

    Arguments:
        n (int, optional): Number of agents (default 1).
        replace (bool, optional):
            Select with replacement (default False).
            If True, the same agent can be selected more than once.

    Returns:
        AgentIter: The selected agents.
    """
    if n == 1:
        selection = [gen.choice(obj_list)]
    elif replace is False:
        selection = gen.sample(obj_list, k=n)
    else:
        selection = gen.choices(obj_list, k=n)
    return AgentIter(model, selection)


class AgentList(AgentSequence, list):
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
        **kwargs:
            Keyword arguments are forwarded
            to the constructor of the new objects.
            Keyword arguments with sequences of type :class:`AttrIter` will be
            broadcasted, meaning that the first value will be assigned
            to the first object, the second to the second, and so forth.
            Otherwise, the same value will be assigned to all objects.

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

    def __add__(self, other):
        agents = AgentList(self.model, self)
        agents.extend(other)
        return agents

    def select(self, selection):
        """ Returns a new :class:`AgentList` based on `selection`.

        Arguments:
            selection (list of bool): List with same length as the agent list.
                Positions that return True will be selected.
        """
        return AgentList(self.model, [a for a, s in zip(self, selection) if s])

    def random(self, n=1, replace=False):
        """ Creates a random sample of agents.

        Arguments:
            n (int, optional): Number of agents (default 1).
            replace (bool, optional):
                Select with replacement (default False).
                If True, the same agent can be selected more than once.

        Returns:
            AgentIter: The selected agents.
        """
        return _random(self.model, self.model.random, self, n, replace)

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


class AgentDList(AgentSequence, ListDict):
    """ Ordered collection of agentpy objects.
    This container behaves similar to :class:`AgentList` in most aspects,
    but comes with additional features for object removal and lookup.

    The key differences to :class:`AgentList` are the following:

    - Faster removal of objects.
    - Faster lookup if object is part of group.
    - No duplicates are allowed.
    - The order of agents in the group cannot be changed.
    - Removal of agents changes the order of the group.
    - :func:`AgentDList.buffer` makes it possible to
      remove objects from the group while iterating over the group.
    - :func:`AgentDList.shuffle` returns an iterator
      instead of shuffling in-place.

    Arguments:
        model (Model): The model instance.
        objs (int or Sequence, optional):
            An integer number of new objects to be created,
            or a sequence of existing objects (default empty).
        cls (type, optional): Class for the creation of new objects.
        **kwargs:
            Keyword arguments are forwarded
            to the constructor of the new objects.
            Keyword arguments with sequences of type :class:`AttrIter` will be
            broadcasted, meaning that the first value will be assigned
            to the first object, the second to the second, and so forth.
            Otherwise, the same value will be assigned to all objects.

    """

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

    def __setattr__(self, name, value):
        if isinstance(value, AttrIter):
            # Apply each value to each agent
            for obj, v in zip(self, value):
                setattr(obj, name, v)
        else:
            # Apply single value to all agents
            for obj in self:
                setattr(obj, name, value)

    def __add__(self, other):
        agents = AgentDList(self.model, self)
        agents.extend(other)
        return agents

    def random(self, n=1, replace=False):
        """ Creates a random sample of agents.

        Arguments:
            n (int, optional): Number of agents (default 1).
            replace (bool, optional):
                Select with replacement (default False).
                If True, the same agent can be selected more than once.

        Returns:
            AgentIter: The selected agents.
        """
        return _random(self.model, self.model.random, self.items, n, replace)

    def select(self, selection):
        """ Returns a new :class:`AgentList` based on `selection`.

        Arguments:
            selection (list of bool): List with same length as the agent list.
                Positions that return True will be selected.
        """
        return AgentList(
            self.model, [a for a, s in zip(self.items, selection) if s])

    def sort(self, var_key, reverse=False):
        """ Returns a new sorted :class:`AgentList`.

        Arguments:
            var_key (str): Attribute of the lists' objects, based on which
                the list will be sorted from lowest value to highest.
            reverse (bool, optional): Reverse sorting (default False).
        """
        agentlist = AgentList(self.model, self)
        agentlist.sort(var_key=var_key, reverse=reverse)
        return agentlist

    def shuffle(self):
        """ Return :class:`AgentIter` over the content of the group
         with the order of objects being shuffled. """
        return AgentDListIter(self.model, self, shuffle=True)

    def buffer(self):
        """ Return :class:`AgentIter` over the content of the group
         that supports deletion of objects from the group during iteration. """
        return AgentDListIter(self.model, self, buffer=True)


class AgentSet(AgentSequence, set):
    """ Unordered collection of agentpy objects.

    Arguments:
        model (Model): The model instance.
        objs (int or Sequence, optional):
            An integer number of new objects to be created,
            or a sequence of existing objects (default empty).
        cls (type, optional): Class for the creation of new objects.
        **kwargs:
            Keyword arguments are forwarded
            to the constructor of the new objects.
            Keyword arguments with sequences of type :class:`AttrIter` will be
            broadcasted, meaning that the first value will be assigned
            to the first object, the second to the second, and so forth.
            Otherwise, the same value will be assigned to all objects.
    """

    def __init__(self, model, objs=(), cls=None, *args, **kwargs):
        if isinstance(objs, int):
            objs = self._obj_gen(model, objs, cls, *args, **kwargs)
        super().__init__(objs)
        super().__setattr__('model', model)
        super().__setattr__('ndim', 1)


class AgentIter(AgentSequence):
    """ Iterator over agentpy objects. """

    def __init__(self, model, source=()):
        object.__setattr__(self, '_model', model)
        object.__setattr__(self, '_source', source)

    def __getitem__(self, item):
        raise AgentpyError(
            'AgentIter has to be converted to list for item lookup.')

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

    def to_list(self):
        """Returns an :class:`AgentList` of the iterator. """
        return AgentList(self._model, self)

    def to_dlist(self):
        """Returns an :class:`AgentDList` of the iterator. """
        return AgentDList(self._model, self)


class AgentDListIter(AgentIter):
    """ Iterator over agentpy objects in an :class:`AgentDList`. """

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
        return self

    def shuffle(self):
        object.__setattr__(self, '_shuffle', True)
        return self

    def _buffered_iter(self):
        """ Iterate over source. """
        items = self._source.items.copy()
        if self._shuffle:
            self._model.random.shuffle(items)
        for a in items:
            if a in self._source:
                yield a
