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

From every level, the following properties can be used to access different parts of the model:

- ``.model`` returns the model instance
- ``.p`` returns an :class:`AttrDict` of the models' parameters
- ``.envs`` returns an :class:`EnvDict` of the objects' environments
- ``.agents`` returns an :class:`AgentList` of the objects' agents (not for :class:`Agent`)
- ``.log`` returns a :class:`dict` of the objects' recorded variables

All of the framework classes are designed to be customized through the creation of
`sub-classes <https://docs.python.org/3/tutorial/classes.html?highlight=inheritance#inheritance>`_
with their own variables and methods. A custom agent type could be defined as follows::

    class my_agent_type(ap.Agent):

        def setup(self):

            """ Called automatically at the agents' creation """

            # Initialize a variable with a parameter
            self.agent_attribute = self.p.my_parameter

        def agent_method(self):

            ...

Some special method-names like ``setup()`` will be used automatically, if declared.
Here is how a basic agent-based model with it's main special method-names could look like::

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

In the demonstration model :doc:`Agentpy_Wealth_Transfer`, you can find a very similar model structure in action.
To use such a model, we have to initialize it with a dictionary of parameters::

    parameters = {'my_parameter':42,
                  'agents':10,
                  'steps':10, }

    model = my_model(parameters)

The parameter ``steps`` will be interpreted as the maximum simulation length.
Other options to control the length of a simulation are:
 
- Defining a method with the name :func:`Model.stop_if`.
- Calling :func:`Model.stop` during the simulation.

To perform a simulation, we can then use the function :func:`Model.run`::

    results = model.run()

Multi-Run Experiments
#####################

The class :class:`Experiment` can be used to run a model multiple times with varied parameters and distinct scenarios.
To prepare a sample of parameters for an experiment, one can use one of the
sampling functions :func:`sample`, :func:`sample_saltelli`, or :func:`sample_discrete`.
In the following example, one parameter is kept fixed while the other two are varied between 10 and 20::

    parameter_ranges = {'my_parameter':42,
                        'agents':(10,20),
                        'steps':(10,20), }

    sample = ap.sample(parameter_ranges, N=10)

An experiment with multiple iterations, scenarios, and parameters can then performed as follows::

    experiment = ap.Experiment(my_model, sample,
                               scenarios=('sc1','sc2'),
                               iterations=10, )

    results = experiment.run()

The demonstration models :doc:`Agentpy_Virus_Spread`, :doc:`Agentpy_Button_Network`, and :doc:`Agentpy_Forest_Fire`
show how such experiments can be used in practice.

Output and Analysis
###################

Both :class:`Model` and :class:`Experiment` can be used to run a simulation,
which will return a :class:`DataDict` with output data.
The output can contain the following categories of data:

- ``log`` holds meta-data about the model and simulation performance.
- ``parameters`` holds the parameter values that have been used for the experiment.
- ``variables`` holds dynamic variables, which can be recorded at multiple time-steps. 
- ``measures`` holds evaluation measures that are recoreded only once per simulation.

.. Agents are identified by a unique ID (:class:`int`), while environments have unique keys (:class:`str`). A model can also be passed to the functions :func:`interactive` and :func:`animate` to generate interactive or animated output. :func:`sensitivity` can be used to analyze the sensitivity of varied parameters.

