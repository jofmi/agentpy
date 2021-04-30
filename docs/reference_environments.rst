.. currentmodule:: agentpy

============
Environments
============

Environments are objects in which agents can inhabit a specific position.
The connection between positions is defined by the environments
topology. There are currently three types:

- :class:`Grid` n-dimensional spatial topology with discrete positions.
- :class:`Space` n-dimensional spatial topology with continuous positions.
- :class:`Network` graph topology consisting of vertices and edges.

----

All three environment classes contain the following methods:

- :func:`add_agents` adds agents to the environment.
- :func:`remove_agents` removes agents from the environment.
- :func:`move_agent` changes an agent's position.
- :func:`neighbors` returns an agent's neighbors within a given distance.

There are further two different types of agents:

- :class:`Agent` can be part of zero or one environment.
- :class:`MultiAgent` can be part of multiple environments.

Both agent classes share the following attributes and methods
for agent-environment interaction:

- :obj:`env` returns the agent's environment(s).
- :obj:`pos` returns the agent's position(s) in their environment(s).
- :func:`move_to` changes the agent's absolute position.
- :func:`move_by` changes the agent's relative position.
- :func:`neighbors` returns the agent's neighbors within a given distance.

.. toctree::
   :hidden:

   reference_grid
   reference_space
   reference_network