.. currentmodule:: agentpy

=========
Sequences
=========

This module offers various data structures to create and manage groups
of both agents and environments. Which structure best to use
depends on the specific requirements of each model.

- :class:`AgentList` provides an enhanced list with
  methods to select and manipulate its entries.
- :class:`AgentGroup` offers additional features
  and better performance for removing and looking up objects.
- :class:`AgentSet` is a simple unordered collection of agents.
- :class:`AgentIter` and :class:`AgentGroupIter` are a list-like iterators
  over a selection of agentpy objects.
- :class:`AttrIter` is a list-like iterator over the attributes of
  each agent in a selection of agentpy objects.

All of these sequence classes can access and manipulate
the methods and variables of their objects as an attribute of the container.
For examples, see :class:`AgentList`.

Containers
##########

.. autoclass:: AgentList
    :members:

.. autoclass:: AgentGroup
    :members:

.. autoclass:: AgentSet
    :members:

Iterators
#########

.. autoclass:: AgentIter
    :members:

.. autoclass:: AgentGroupIter
    :members:

.. autoclass:: AttrIter
    :members:
