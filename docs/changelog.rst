.. currentmodule:: agentpy

=========
Changelog
=========

0.1.0.dev
---------

This update contains some major revisions of most classes and methods in the
package, including new features, better performance, and a more coherent syntax.
Unfortunately, this causes some API changes without backward compatibility.
There will hopefully be less major changes from now on,
and backward compatibility will be ensured after first major release (v1.0.0).

Object creation
...............

The methods :func:`add_agents`, :func:`add_env`, etc. have been depreciated.
Instead, new objects can now be created directly or through sequences.
For example::

    class Model(ap.Model):
        def setup(self):
            self.special_agent = ap.Agent()  # Create a single agent
            self.agents = ap.AgentList(self, 10)  # Create a sequence of 10 agents
            self.grid = ap.Grid(self, (5, 5))  # Create a grid environment

Data structures
...............

The main advantage of the new way of object creation is increased control
over data structures in which to hold objects. Some agents can be kept in
different sequences than others, and different kinds of sequences can be used.

In addition to :class:`AgentList`, there is new sequence type
:class:`AgentDList` that provides increased performance
for the lookup and deletion of agents.
It also comes with a method `buffer` that allows for deletion of agents
from the group during an active iteration over the group.

The structure of :class:`AttrList` has been changed to an iterable over its
source list. This improves performance and makes it possible to change
agent attributes by setting new values to items in the attribute list (see
:class:`AgentList` for an example). Otherwise, the class behaves as before.

Environments
............

The agent class has been split into two types :class:`Agent` and :class:`MultiAgent`,
and can access their environment(s) through the attributes `pos` and `env`.
The former should be used for models with zero or one environment, while the latter
can be part of multiple environments.

All three environment types have been revised.
:class:`Grid` is now a structured array
that can hold field attributes per position in addition to agents,
and can be customized with the arguments `torus`, `track_free`, and `check_border`.
:class:`Network` can now hold multiple agents per node.

Random number generators
........................

:class:`Model` now contains two random number generators `random` and `nprandom`
so that both standard and numpy random operations can be used.
The parameter `seed` can be used to initialize both.
:class:`Experiment` now also has an argument `random` to control whether
to vary seeds over different iterations.

Parameter samples
.................

Variable parameters can now be defined with the two new classes
:class:`Range` (for continuous parameter ranges) and :class:`Values` (for pre-defined of discrete parameter values).
Parameter dictionaries with these classes can be used to create samples,
but can also be passed to a normal model, which will then use default values.

The sampling methods :func:`sample`, :func:`sample_discrete`, and :func:`sample_saltelli`
have been depreciated and integrated into the new class :class:`Sample`,
which comes with additional features to create new kinds of samples.

Data analysis
.............

The structure of output data from experiments has been changed.
The name of `measures` has been changed to `reporters`.
Parameters are now stored in the two categories `constants` and `sample`.
Variables are stored in separate dataframes based on the object type.
The dataframes' index is now separatedinto `sample_id` and `iteration`.

:class:`DataDict` now contains new methods.

0.0.7 (March 2021)
------------------

Continuous space environments
.............................

A new environment type :class:`Space` and method :func:`Model.add_space`
for agent-based models with continuous space topologies has been added.
There is a new demonstration model :doc:`agentpy_flocking` in the model library,
which shows how to simulate the flocking behavior of animals
and demonstrates the use of the continuous space environment.

Random generators
.................

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
