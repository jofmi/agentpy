.. currentmodule:: agentpy

=========
Reference
=========

Overview
########

The main framework for agent-based models consists of four levels:

- :class:`experiment`, which holds a model, parameter sample, & settings
- :class:`model`, which holds agents, environments, parameters, & procedures
- :class:`environment` and :class:`network`, which hold agents
- :class:`agent`

Agents are identified by a unique ID (str), while environments have unique keys (str). Groups of agents and environments are held within the classes :class:`agent_list` and :class:`env_dict`. 

Both :class:`model` and :class:`experiment` can be used to run a simulation, which will yield a :class:`data_dict` with output data. The function :func:`sample` can be used to create parameter samples for an experiment, and :func:`sensitivity` can be used to analyze the sensitivity of varied parameters.

In addition to the experiments, a model can also be passed to the functions :func:`interactive` and :func:`animate` to generate interactive or animated output.

To design a custom model, these classes can be used as parent classes and be expanded with custom attributes and methods. See :doc:`tutorials` for model examples.

Model framework
###############

Agents
------

.. autoclass:: agent
   :members:

.. autoclass:: agent_list
   :members:

Environments
------------

.. autoclass:: environment
   :members:

.. autoclass:: network
   :members:

.. autoclass:: env_dict
   :members:

Agent-based models
------------------

.. autoclass:: model
   :members:

Experiments
###########

Parameter sampling
------------------

.. autofunction:: sample

Experiment class
----------------

.. autoclass:: experiment
   :members:

.. class:: exp()

   Alias of :class:`experiment`

Output & Analysis
#################

Output data
-----------

.. autoclass:: data_dict
   :members:

Sensitivity analysis
--------------------

.. autofunction:: sensitivity

Interactive output
------------------

.. autofunction:: interactive

Animations
----------

.. autofunction:: animate


Base classes
############

.. autoclass:: attr_dict

.. autoclass:: attr_list