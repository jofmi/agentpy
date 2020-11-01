.. currentmodule:: agentpy

========
Overview
========

Throughout this documentation, agentpy is assumed to be imported as follows::

    import agentpy as ap

Agent-based Models
##################

The basic framework for agent-based models consists of three levels:

1. :class:`Model`, which contains agents, environments, parameters, & procedures
2. :class:`Environment`, :class:`Grid`, and :class:`Network`, which contain agents
3. :class:`Agent`, the basic building blocks of the model

All of the framework classes are designed to be customized through the creation of `sub-classes <https://docs.python.org/3/tutorial/classes.html?highlight=inheritance#inheritance>`_ with their own variables and methods. A custom agent type could be defined as follows::

    class my_agent_type(ap.Agent):

        def setup(self):

            """ Called automatically at the agents' creation """

            self.agent_attribute = 0 # Initialize a variable

        def agent_method(self):

            ...

Some methods with special names like :func:`agent.setup` will be used automatically if they are defined. There are further some standard properties that can be used to access different parts of the model from within each object:

- ``.model`` returns the model instance
- ``.p`` returns an :class:`AttrDict` of the models' parameters
- ``.envs`` returns an :class:`EnvDict` of the objects' environments
- ``.agents`` returns an :class:`AgentList` of the objects' agents (not for :class:`Agent`)
- ``.log`` returns a :class:`dict` of the objects' recorded variables

Here how a basic agent-based model and it's main special methods could look like::

    class my_model(ap.Model):

        def setup(self):

            """ Called at the start of the simulation """

            # Add new agents, with the number given by a parameter
            self.add_agents(self.p.agents, my_agent_type)

        def step(self):

            """ Called at every simulation step """

            # Call a method for every agent
            self.agents.agent_method()

        def update(self):

            """ Called after setup as well as after each step """

            # Record a dynamic agent variable (once per time-step)
            self.agents.record('agent_attribute')

        def end(self):

            """ Called at the end of the simulation """

            # Record a measure (once per simulation)
            self.measure('my_measure', 1)

To use this model, we have to initialize it with a set of parameters::

    parameters = {'agents':10, 'steps':10}
    model = my_model(parameters)

The parameter steps as a parameter ``steps`` will be used automatically to define how long the simulation will run. Other options to define the length of a simulation are:
 
- Defining a custom stop condition with :func:`Model.stop_if`.
- Calling :func:`Model.stop` during the simulation.

To perform a simulation, we can then use the function :func:`Model.run`::

    results = model.run()

Multi-Run Experiments
#####################

The class :class:`Experiment` can be used to run a model multiple times with varied parameters and distinct scenarios. To prepare a sample of parameters for an experiment, one can use one of the sampling functions, like for example::

    parameter_ranges = {'agents':(10,20);'steps':(10,20)}
    sample = ap.sample(parameter_ranges, N = 10 )

An experiment with multiple iterations, scenarios, and parameters is performed as follows::

    experiment = ap.Experiment(my_model, sample,
                               scenarios=('sc1','sc2'),
                               iterations=10, )

    results = experiment.run()

Output and Analysis
###################

Both :class:`Model` and :class:`Experiment` can be used to run a simulation, which will yield a :class:`DataDict` with output data. The output can contain the following categories of data:

- ``log`` holds meta-data about the model and simulation performance.
- ``parameters`` holds the parameter values that have been used for this simulation.
- ``variables`` holds dynamic variables, which can be recorded at multiple time-steps. 
- ``measures`` holds evaluation measures that are recoreded only once per simulation.

.. Agents are identified by a unique ID (:class:`int`), while environments have unique keys (:class:`str`). A model can also be passed to the functions :func:`interactive` and :func:`animate` to generate interactive or animated output. :func:`sensitivity` can be used to analyze the sensitivity of varied parameters.

