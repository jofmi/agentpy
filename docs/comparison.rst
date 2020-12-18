.. currentmodule:: agentpy

==========
Comparison
==========

Agentpy vs. Mesa
################

An alternative framework for agent-based modeling in Python is `Mesa <https://mesa.readthedocs.io/>`__.
The stated goal of Mesa is *"to be the Python 3-based counterpart to NetLogo, Repast, or MASON"*.
The focus of these frameworks is traditionally on spatial environments, with an interface where one can
observe live dynamics and adjust parameters while the model is running.

Agentpy, in contrast, is more focused on networks and :ref:`multi-run experiments<overview_experiments>`,
with tools to generate and analyze :ref:`output data<overview_output>` from these experiments.
Agentpy further has a different model structure
that is built around :ref:`agent lists <overview_agents>`,
which allow for simple selection and manipulation of agent groups;
and :ref:`environments <overview_environments>`,
which can contain agents but also act as agents themselves.

To allow for an comparison of the syntax of each framework,
here are two examples for a simple model of wealth transfer,
both of which realize exactly the same operations.
More information on the two models can be found in the documentation
of each framework (link for :doc:`Agentpy <agentpy_wealth_transfer>` &
`Mesa <https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#tutorial-description>`_).

+--------------------------------------------+----------------------------------------------+
|**Agentpy**                                 |**Mesa**                                      |
+--------------------------------------------+----------------------------------------------+
|                                            |                                              |
|.. literalinclude:: agentpy_demo.py         |.. literalinclude:: mesa_demo.py              |
|                                            |                                              |
+--------------------------------------------+----------------------------------------------+

Finally, the following table provides a comparison of the main features of each framework.

==========================  ===================================  ===================================
**Feature**                 **Agentpy**                          **Mesa**
| Customizable objects      | Agent, Environment, Model	         | Agent, Model
| Container classes         | AgentList and EnvDict for          | Scheduler (see below)
                            | selection and manipulation
                            | of agent and environment groups
| Time management           | Custom activation order has to be  | Multiple scheduler classes for
                            | defined in the Model.step method   | different activation orders
| Supported topologies      | Spatial grid, networkx graph       | Spatial grid, network grid,
                            |                                    | continuous space
| Data recording            | Recording methods for variables    | DataCollector class that can
                            | (of agents, environments, and      | collect variables of agents
                            | model) and evaluation measures     | and model
| Parameter sampling        | Multiple sampling functions        | Custom sample has to be defined
| Multi-run experiments     | Experiment class that supports     | BatchRunner class that supports
                            | multiple iterations, parameter     | multiple iterations and parameter
                            | samples, scenario comparison,      | samples
                            | and parallel processing
| Output data               | DataDict class that can save,      | Multiple methods to generate
                            | load, and re-arrange output data   | dataframes
| Visualization             | Tools for plots, animations,       | Extensive browser-based
                            | and interactive visualization in   | visualization module
                            | Python
| Analysis                  | Tools for data arrangement and
                            | sensitivity analysis
==========================  ===================================  ===================================