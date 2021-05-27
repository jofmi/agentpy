.. currentmodule:: agentpy

======
Agents
======

Agent-based models can contain multiple agents of different types.
This module provides a base class :class:`Agent`
that is meant to be used as a template to create custom agent types.
Initial variables should be defined by overriding :func:`Agent.setup`.

.. autoclass:: Agent
    :members:
    :inherited-members: