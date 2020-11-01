.. currentmodule:: agentpy

=============
API Reference
=============

Agents
######

.. autoclass:: Agent
   :members:

.. autoclass:: AgentList
   :members:

Environments
############

.. autoclass:: Environment
   :members:

.. autoclass:: EnvDict
   :members:

Networks
--------

.. autoclass:: Network
   :members:

Spacial grids
-------------

.. autoclass:: Grid
   :members:


Agent-based models
##################

.. autoclass:: Model
   :members:


Parameter sampling
##################

.. autofunction:: sample

.. autofunction:: sample_discrete

.. autofunction:: sample_saltelli


Experiments
###########

.. class:: Exp()

   Alias of :class:`Experiment`

.. autoclass:: Experiment
   :members:


Output data
###########

.. autoclass:: DataDict
   :members:

Analysis
########

Interactive output
------------------

.. autofunction:: interactive


Sobol sensitivity
-----------------

.. autofunction:: sobol_sensitivity

Animations
----------

.. autofunction:: animate

Plots
-----

.. autofunction:: gridplot


Base classes
############

.. autoclass:: AttrDict

.. autoclass:: ObjList