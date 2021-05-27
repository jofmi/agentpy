.. currentmodule:: agentpy

============
Environments
============

Environments are objects in which agents can inhabit a specific position.
The connection between positions is defined by the environment's
topology. There are currently three types:

- :class:`Grid` n-dimensional spatial topology with discrete positions.
- :class:`Space` n-dimensional spatial topology with continuous positions.
- :class:`Network` graph topology consisting of :class:`AgentNode` and edges.

All three environment classes contain the following methods:

- :func:`add_agents` adds agents to the environment.
- :func:`remove_agents` removes agents from the environment.
- :func:`move_to` changes an agent's position.
- :func:`move_by` changes an agent's position, relative to their current position.
- :func:`neighbors` returns an agent's neighbors within a given distance.

.. toctree::
   :hidden:

   reference_grid
   reference_space
   reference_network