.. currentmodule:: agentpy

=========
Changelog
=========

0.0.6.dev
---------

* New demonstration model :doc:`agentpy_segregation`.
  Other demonstrations have been updated.
* All model objects now have a unique id number of type :class:`int`.
  Methods that take an agent or environment as an argument
  can now take either the instance or id of the object.
  The :attr:`key` attribute of environments has been removed.
* :func:`Model.run` now takes an optional argument `steps`.
* :class:`EnvDict` has been replaced by :class:`EnvList`,
  which has the same functionalities as :class:`AgentList`.
* Model objects now have a property :attr:`env`
  that returns the first environment of the object.
* Argument `map_to_nodes` has been removed from :func:`Network.add_agents`.
  Instead, agents can be mapped to nodes by passing an AgentList to the agents argument of :func:`Model.add_network`.
* Revised methods for :class:`Grid`:

  * :func:`Agent.move_to` and :func:`Agent.move_by` can be used to move agents.
  * :func:`Grid.items` returns an iterator of position and agent tuples.
  * :func:`Grid.get_agents` returns agents in selected position or area.
  * :func:`Grid.position` returns the position coordinates for an agent.
  * :func:`Grid.positions` returns an iterator of position coordinates.
  * :func:`Grid.attribute` returns a nested list with values of agent attributes.
  * :func:`Grid.apply` returns nested list with return values of a custom function.

* :func:`gridplot` now takes a grid of values as an input and can convert them to rgba.
* :func:`animate` now takes a model instance as an input instead of a class and parameters.
* :func:`sample` and :func:`sample_saltelli` will now return integer values for parameters
  if parameter ranges are given as integers. For float values,
  a new argument `digits` can be passed to round parameter values.
* The function :func:`interactive` has been removed, and is replaced by the
  new method :func:`Experiment.interactive`.
* :func:`sobol_sensitivity` has been changed to :func:`sensitivity_sobol`.

0.0.5 (December 2020)
---------------------

* :func:`Experiment.run` now supports parallel processing.
* New methods :func:`DataDict.arrange_variables` and :func:`DataDict.arrange_measures`,
  which generate a dataframe of recorded variables or measures and varied parameters.
* Major revision of :func:`DataDict.arrange`, see new description in the documentation.
* New features for :class:`AgentList`: Arithmethic operators can now be used with :class:`AttrList`.

0.0.4 (November 2020)
---------------------

* First major release.
