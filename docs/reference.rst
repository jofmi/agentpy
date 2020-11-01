.. currentmodule:: agentpy

=============
API Reference
=============

Agents
######

.. autoclass:: Agent
    :members:
    :inherited-members:

.. autoclass:: AgentList
    :members:
    :inherited-members:

Environments
############

.. autoclass:: Environment
     :members:
     :inherited-members:

.. autoclass:: EnvDict
    :members:
    :inherited-members:

Networks
--------

.. autoclass:: Network
     :members:
     :inherited-members:

Spacial grids
-------------

.. autoclass:: Grid
    :members:
    :inherited-members:


Agent-based models
##################

.. autoclass:: Model
    :members:
    :inherited-members:


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
    :inherited-members:


Output data
###########

.. autoclass:: DataDict
    :members:
    :inherited-members:

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