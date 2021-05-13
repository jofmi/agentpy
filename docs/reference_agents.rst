.. currentmodule:: agentpy

======
Agents
======

Agent-based models can contain multiple agents of different types.
This module provides two classes :class:`Agent` and :class:`MultiAgent`
that are meant to be used as a template to create custom agent types.
Initial variables can by overriding :func:`Agent.setup`.

Standard agents
###############

.. autoclass:: Agent
    :members:
    :inherited-members:

Multi-environment agents
########################

.. autoclass:: MultiAgent
    :members: