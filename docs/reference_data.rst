.. currentmodule:: agentpy

=============
Data analysis
=============

This module offers tools to access, arrange, analyse, and store output data from simulations.
A :class:`DataDict` can be generated by the methods :func:`Model.run`, :func:`Experiment.run`, and :func:`DataDict.load`.

.. autoclass:: DataDict

Data arrangement
################

.. automethod:: DataDict.arrange
.. automethod:: DataDict.arrange_reporters
.. automethod:: DataDict.arrange_variables

Analysis methods
################

.. automethod:: DataDict.calc_sobol

Save and load
#############

.. automethod:: DataDict.save
.. automethod:: DataDict.load