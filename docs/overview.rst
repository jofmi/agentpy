.. currentmodule:: agentpy

========
Overview
========

This section aims to provide a rough overview over the main classes and
functions of agentpy and how they are meant to be used.
For a more detailed description of each element, please refer to the :doc:`reference`.
Throughout this documentation, agentpy is imported as follows::

    import agentpy as ap

Creating models
###############

The basic framework for agent-based models consists of three levels:

1. :class:`Model`, which contains agents, environments, parameters, & procedures
2. :class:`Environment`, :class:`Grid`, and :class:`Network`, which contain agents
3. :class:`Agent`, the basic building blocks of the model

All of these classes are designed to be customized through the creation of
`sub-classes <https://docs.python.org/3/tutorial/classes.html?highlight=inheritance#inheritance>`_
with their own variables and methods.
A custom agent type could be defined as follows::

    class MyAgentType(ap.Agent):

        def setup(self):
            # Initialize an attribute with a parameter
            self.my_attribute = self.p.my_parameter

        def agent_method(self):
            # Define custom actions here
            pass

The method :func:`Agent.setup` is meant to be overwritten
and will be called after an agents' creation.
All variables of an agents should be initialized in this method.
Other methods can represent actions that the agent will be able to take during a simulation.

We can further see that the agent comes with a built-in attribute :attr:`.p` that
allows it to access the models' parameters.
All model objects (i.e. agents, environments, and the model itself)
are equipped with such properties to access different parts of the model:

- :attr:`.model` returns the model instance
- :attr:`.model.t` returns the model's time-step
- :attr:`.id` returns a unique identifier number for each object
- :attr:`.p` returns an :class:`AttrDict` of the models' parameters
- :attr:`.envs` returns an :class:`EnvList` of the objects' environments
- :attr:`.agents` (not for agents) returns an :class:`AgentList` of the objects' agents
- :attr:`.log` returns a :class:`dict` of the objects' recorded variables

Using the new agent type defined above,
here is how a basic model could look like::

    class MyModel(ap.Model):

        def setup(self):
            """ Called at the start of the simulation """
            self.add_agents(self.p.agents, MyAgentType)  # Add new agents

        def step(self):
            """ Called at every simulation step """
            self.agents.agent_method()  # Call a method for every agent

        def update(self):
            """ Called after setup as well as after each step """
            self.agents.record('my_attribute')  # Record a dynamic variable

        def end(self):
            """ Called at the end of the simulation """
            self.measure('my_measure', 1)  # Record an evaluation measure

This custom model is defined by four special methods
that will be used automatically during different parts of a simulation.
If you want to see a basic model like this in action,
take a look at the :doc:`agentpy_wealth_transfer` demonstration in the :doc:`model_library`.

.. _overview_agents:

Using agents
############

Agentpy comes with various tools to create, manipulate, and delete agents.
The method :func:`Model.add_agents` can be used to initialize new agents.
A list of all agents in a model can be accessed through :attr:`Model.agents`.
Lists of agents are returned as an :class:`AgentList`,
which provides special features to access and manipulate the whole group of agents.

For example, when the model defined above calls :func:`self.agents.agent_method`,
it will call the method :func:`MyAgentType.agent_method` for every agent in the model.
Similar commands can be used to set and access variables, or select subsets
of agents with boolean operators.
The following command, for example, would select all agents with an id above one::

    self.agents.select(self.agents.id > 1)

Further examples can be found in the :class:`AgentList` reference
or the :doc:`agentpy_virus_spread` model.

.. _overview_environments:

Using environments
##################

Environments can contain agents just like the main model,
and are useful if one wants to regard particular topologies for interaction
or multiple environments that can hold seperate populations of agents.
Agents can be moved between environments with the methods
:func:`Agent.enter`, :func:`Agent.exit`, and :func:`Agent.move`.

New environments can be created with :func:`Model.add_env`.
Similar to agents, the attribute :attr:`envs` returns an :class:`EnvList`
with special features to deal with groups of environments.
There are three different types of environments:

- :class:`Environment`, which simply contain agents without any topology
- :class:`Network`, in which agents can be connected via a networkx graph
- :class:`Grid`, in which agents occupy a position on a x-dimensional space

Applied examples of networks can be found in the demonstration models
:doc:`agentpy_virus_spread` and :doc:`agentpy_button_network`,
while a spatial grid is used in :doc:`agentpy_forest_fire`.

Recording data
##############

As can be seen in the model defined above,
there are two main types of data in agentpy.
The first are dynamic variables,
which can be stored for each object (agent, environment, or model) and time-step.
They are useful to look at the dynamics of individual or aggregate objects over time
and can be recorded by calling the method :meth:`record` for the respective object.

The other type of recordable data are evaluation measures.
These, in contrast, can be stored only for the model as a whole and only once per run.
They are useful as summary statistics that can be compared over multiple runs,
and can be recorded with the method :meth:`Model.measure`.

.. _overview_simulation:

Running a simulation
####################

To perform a simulation, we have to initialize a new instance of our model type
with a dictionary of parameters, after which we use the function :func:`Model.run`.
This will return a :class:`DataDict` with recorded data from the simulation.
A simple run could be prepared and executed as follows::

    parameters = {'my_parameter':42,
                  'agents':10,
                  'steps':10, }

    model = MyModel(parameters)
    results = model.run()

The procedure of a simulation is as follows:

0. The model initializes with the time-step ``Model.t = 0``
1. :func:`Model.setup` and :func:`Model.update` are called
2. The model's time-step is increased by 1
3. :func:`Model.step` and :func:`Model.update` are called
4. Step 2 and 3 are repeated until the simulation is stopped.
5. :func:`Model.end` is called.

The simulation of a model can be stopped by one of the following two ways:

1. Calling the :func:`Model.stop` during the simulation.
2. Reaching the time-limit, which can be passed as an entry `steps` in the
   parameter dictionary or as an argument `steps` to :func:`Model.run`.

.. _overview_experiments:

Multi-run experiments
#####################

The class :class:`Experiment` can be used to run a model multiple times
with repeated iterations, varied parameters, and distinct scenarios.
To prepare a sample of parameters for an experiment, one can use one of the
sampling functions :func:`sample`, :func:`sample_saltelli`, or :func:`sample_discrete`.
Here is an example of an experiment with the model defined above::

    parameter_ranges = {'my_parameter': 42,
                        'agents': (10, 20, int),
                        'steps': (10, 20, int)}

    sample = ap.sample(parameter_ranges, n=5)

    exp = ap.Experiment(MyModel, sample, iterations=2,
                        scenarios=('sc1','sc2'))

    results = exp.run()

In this experiment, we use a sample where one parameter is kept fixed
while the other two are varied 5 times from 10 to 20 and set to integer.
Every possible combination is repeated 2 times, which results in 50 runs.
Each run further has one result for each of the two scenarios `sc1` and `sc2`.
For more applied examples of experiments, check out the demonstration models
:doc:`agentpy_virus_spread`, :doc:`agentpy_button_network`, and :doc:`agentpy_forest_fire`.

.. _overview_output:

Output and analysis
###################

Both :class:`Model` and :class:`Experiment` can be used to run a simulation,
which will return a :class:`DataDict` with output data.
The output from the experiment defined above looks as follows::

    >>> results
    DataDict {
    'log': Dictionary with 5 keys
    'parameters':
        'fixed': Dictionary with 1 key
        'varied': DataFrame with 2 variables and 25 rows
    'measures': DataFrame with 1 variable and 50 rows
    'variables':
        'my_agent_type': DataFrame with 1 variable and 10500 rows
    }

The output can contain the following categories of data:

- :attr:`log` holds meta-data about the model and simulation performance.
- :attr:`parameters` holds the parameter values that have been used for the experiment.
- :attr:`variables` holds dynamic variables, which can be recorded at multiple time-steps.
- :attr:`measures` holds evaluation measures that are recoreded only once per simulation.

This data can be stored with :func:`DataDict.save` and :func:`load`.
:func:`DataDict.arrange` can further be used to generate a specific
dataframe for analysis or visualization. All data is given in a :class:`pandas.DataFrame` and
formatted as `long-form data <https://seaborn.pydata.org/tutorial/data_structure.html>`_,
which makes it compatible to use with statistical packages like `seaborn <https://seaborn.pydata.org/>`_.
Agentpy further provides the following functions for analysis:

- :func:`sensitivity_sobol` performs a Sobol sensitivity analysis.
- :func:`Experiment.interactive` generates an interactive widget for parameter variation.
- :func:`animate` generates an animation that can display output over time.
- :func:`gridplot` visualizes agent positions on a spatial :class:`Grid`.

To see applied examples of these functions, please check out the :doc:`model_library`.