.. currentmodule:: agentpy

==================
Agent-based models
==================

The :class:`Model` contains all objects
and defines the procedures of an agent-based simulation.
It is meant as a template for custom model classes that
override the `custom procedure methods`_.

.. autoclass:: Model

Simulation tools
################

.. automethod:: Model.run
.. automethod:: Model.stop

.. _custom procedure methods:

Custom procedures
#################

.. automethod:: Model.setup
.. automethod:: Model.step
.. automethod:: Model.update
.. automethod:: Model.end

Data collection
###############

.. automethod:: Model.record
.. automethod:: Model.report

Conversion
##########

.. automethod:: Model.as_function