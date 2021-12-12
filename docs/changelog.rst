.. currentmodule:: agentpy

=========
Changelog
=========

0.1.5.dev
---------

- :func:`Model.run` can now continue simulations that have already been run.
  The steps defined in the argument 'steps' now reflect additional steps,
  which will be added to the models current time-step.
  Random number generators will not be re-initialized in this case.
- :func:`animate` has been improved.
  It used to stop the animation one step too early, which has been fixed.
  Two faulty import statements have been corrected.
  And, as above, the argument 'steps' now also reflects additional steps.

0.1.4 (September 2021)
----------------------

- :class:`AttrIter` now returns a new :class:`AttrIter` when called as a function.
- :func:`gridplot` now returns an :class:`matplotlib.image.AxesImage`
- :func:`DataDict.save` now
  supports values of type :class:`numpy.bool_`
  and can re-write to existing directories if an existing `exp_id` is passed.
- :func:`DataDict.load` now supports the argument `exp_id = 0`.
- :func:`animate` now supports more than 100 steps.
- :class:`AttrIter` now returns a new :class:`AttrIter` when called as a function.
- :class:`Model` can take a new parameter `report_seed` (default True) that
  indicates whether the seed of the current run should be reported.

0.1.3 (August 2021)
-------------------

- The :class:`Grid` functionality `track_empty` has been fixed
  to work with multiple agents per cell.
- Getting and setting items in :class:`AttrIter` has been fixed.
- Sequences like :class:`AgentList` and :class:`AgentDList`
  no longer accept `args`, only `kwargs`.
  These keyword arguments are forwarded
  to the constructor of the new objects.
  Keyword arguments with sequences of type :class:`AttrIter` will be
  broadcasted, meaning that the first value will be assigned
  to the first object, the second to the second, and so forth.
  Otherwise, the same value will be assigned to all objects.

0.1.2 (June 2021)
-----------------

- The property :attr:`Network.nodes` now returns an :class:`AttrIter`,
  so that network nodes can be assigned to agents as follows::

      self.nw = ap.Network(self)
      self.agents = ap.AgentList(self, 10)
      self.nw.add_agents(self.agents)
      self.agents.node = self.nw.nodes

- :class:`AgentIter` now requires the model to be passed upon creation
  and has two new methods :func:`AgentIter.to_list` and
  :func:`AgentIter.to_dlist` for conversion between sequence types.
- Syntax highlighting in the documentation has been fixed.

0.1.1 (June 2021)
-----------------

- Marked release for the upcoming JOSS publication of AgentPy.
- Fixed :func:`Grid.move_to`: Agents can now move to their current position.

0.1.0 (May 2021)
----------------

This update contains major revisions of most classes and methods in the
library, including new features, better performance, and a more coherent syntax.
The most important API changes are described below.

Object creation
...............

The methods :func:`add_agents`, :func:`add_env`, etc. have been removed.
Instead, new objects are now created directly or through :doc:`reference_sequences`.
This allows for more control over data structures (see next point) and attribute names.
For example::

    class Model(ap.Model):
        def setup(self):
            self.single_agent = ap.Agent()  # Create a single agent
            self.agents = ap.AgentList(self, 10)  # Create a sequence of 10 agents
            self.grid = ap.Grid(self, (5, 5))  # Create a grid environment

Data structures
...............

The new way of object creation makes it possible to choose specific data structures for different groups of agents.
In addition to :class:`AgentList`, there is a new sequence type
:class:`AgentDList` that provides increased performance
for the lookup and deletion of agents.
It also comes with a method :func:`AgentDList.buffer`
that allows for safe deletion of agents
from the list while it is iterated over

:class:`AttrList` has been replaced by :class:`AttrIter`.
This improves performance and makes it possible to change
agent attributes by setting new values to items in the attribute list (see
:class:`AgentList` for an example). In most other ways, the class still behaves like a normal list.
There are also two new classes :class:`AgentIter` and :class:`AgentDListIter` that are returned by some of the library's methods.

Environments
............

The three environment classes have undergone a major revision.
The :func:`add_agents` functions have been extended with new features
and are now more consistent between the three environment classes.
The method :func:`move_agents` has been replaced by :func:`move_to` and :func:`move_by`.
:class:`Grid` is now defined as a structured numpy array
that can hold field attributes per position in addition to agents,
and can be customized with the arguments `torus`, `track_empty`, and `check_border`.
:func:`gridplot` has been adapted to support this new numpy structure.
:class:`Network` now consists of :class:`AgentNode` nodes that can hold multiple agents per node, as well as node attributes.

Environment-agent interaction
.............................

The agents' `env` attribute has been removed.
Instead, environments are manually added as agent attributes,
giving more control over the attribute name in the case of multiple environments.
For example, agents in an environment can be set up as follows::

    class Model(ap.Model):
        def setup(self):
            self.agents = ap.AgentList(self, 10)
            self.grid = self.agents.mygrid = ap.Grid(self, (10, 10))
            self.grid.add_agents(self.agents)

The agent methods `move_to`, `move_by`, and `neighbors` have also been removed.
Instead, agents can access these methods through their environment.
In the above example, a given agent `a` could for example access their position
through `a.mygrid.positions[a]` or their neighbors through calling `a.mygrid.neighbors(a)`.

Parameter samples
.................

Variable parameters can now be defined with the three new classes
:class:`Range` (for continuous parameter ranges), :class:`IntRange` (for integer parameter ranges), and :class:`Values` (for pre-defined of discrete parameter values).
Parameter dictionaries with these classes can be used to create samples,
but can also be passed to a normal model, which will then use default values.
The sampling methods :func:`sample`, :func:`sample_discrete`, and :func:`sample_saltelli`
have been removed and integrated into the new class :class:`Sample`,
which comes with additional features to create new kinds of samples.

Random number generators
........................

:class:`Model` now contains two random number generators `Model.random` and `Model.nprandom`
so that both standard and numpy random operations can be used.
The parameter `seed` can be used to initialize both generators.
:class:`Sample` has an argument `randomize` to vary seeds over parameter samples.
And :class:`Experiment` has a new argument `randomize` to control whether
to vary seeds over different iterations.
More on this can be found in :doc:`guide_random`.

Data analysis
.............

The structure of output data in :class:`DataDict` has been changed.
The name of `measures` has been changed to `reporters`.
Parameters are now stored in the two categories `constants` and `sample`.
Variables are stored in separate dataframes based on the object type.
The dataframe's index is now separated into `sample_id` and `iteration`.
The function :func:`sensitivity_sobol` has been removed and is replaced
by the method :func:`DataDict.calc_sobol`.

Interactive simulations
.......................

The method :func:`Experiment.interactive` has been removed and is replaced
by an interactive simulation interface that is being developed in the separate
package `ipysimulate <https://github.com/JoelForamitti/ipysimulate>`_.
This new package provides interactive javascript widgets with parameter sliders
and live plots similar to the traditional NetLogo interface.
Examples can be found in :doc:`guide_interactive`.

0.0.7 (March 2021)
------------------

Continuous space environments
.............................

A new environment type :class:`Space` and method :func:`Model.add_space`
for agent-based models with continuous space topologies has been added.
There is a new demonstration model :doc:`agentpy_flocking` in the model library,
which shows how to simulate the flocking behavior of animals
and demonstrates the use of the continuous space environment.

Random number generators
........................

:class:`Model` has a new property :obj:`Model.random`, which returns the
models' random number generator of type :func:`numpy.random.Generator`.
A custom seed can be set for :func:`Model.run` and :func:`animate`
by either passing an argument or defining a parameter :attr:`seed`.
All methods with stochastic elements like :func:`AgentList.shuffle`
or :func:`AgentList.random` now take an optional argument `generator`,
with the model's main generator being used if none is passed.
The function :func:`AgentList.random` now uses :func:`numpy.random.Generator.choice`
and has three new arguments 'replace', 'weights', and 'shuffle'.
More information with examples can be found in the API reference
and the new user guide :doc:`guide_random`.

Other changes
.............

* The function :func:`sensitivity_sobol` now has an argument :attr:`calc_second_order` (default False).
  If True, the function will add second-order indices to the output.
* The default value of :attr:`calc_second_order` in :func:`sample_saltelli`
  has also been changed to False for consistency.
* For consistency with :class:`Space`,
  :class:`Grid` no longer takes an integer as argument for 'shape'.
  A tuple with the lengths of each spatial dimension has to be passed.
* The argument 'agents' has been removed from :class:`Environment`.
  Agents have to be added through :func:`Environment.add_agents`.

Fixes
.....

* The step limit in :func:`animate` is now the same as in :func:`Model.run`.
* A false error message in :func:`DataDict.save` has been removed.

0.0.6 (January 2021)
--------------------

* A new demonstration model :doc:`agentpy_segregation` has been added.
* All model objects now have a unique id number of type :class:`int`.
  Methods that take an agent or environment as an argument
  can now take either the instance or id of the object.
  The :attr:`key` attribute of environments has been removed.
* Extra keyword arguments to :class:`Model` and :class:`Experiment`
  are now forwarded to :func:`Model.setup`.
* :func:`Model.run` now takes an optional argument `steps`.
* :class:`EnvDict` has been replaced by :class:`EnvList`,
  which has the same functionalities as :class:`AgentList`.
* Model objects now have a property :attr:`env`
  that returns the first environment of the object.
* Revision of :class:`Network`.
  The argument `map_to_nodes` has been removed from :func:`Network.add_agents`.
  Instead, agents can be mapped to nodes by passing an AgentList to the agents argument of :func:`Model.add_network`.
  Direct forwarding of attribute calls to :attr:`Network.graph` has been
  removed to avoid confusion.
* New and revised methods for :class:`Grid`:

  * :func:`Agent.move_to` and :func:`Agent.move_by` can be used to move agents.
  * :func:`Grid.items` returns an iterator of position and agent tuples.
  * :func:`Grid.get_agents` returns agents in selected position or area.
  * :func:`Grid.position` returns the position coordinates for an agent.
  * :func:`Grid.positions` returns an iterator of position coordinates.
  * :func:`Grid.attribute` returns a nested list with values of agent attributes.
  * :func:`Grid.apply` returns nested list with return values of a custom function.
  * :func:`Grid.neighbors` has new arguments `diagonal` and `distance`.

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

First documented release.
