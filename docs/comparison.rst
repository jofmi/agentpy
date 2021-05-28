.. currentmodule:: agentpy

==========
Comparison
==========

There are numerous modeling and simulation tools for ABMs,
each with their own particular focus and style
(find an overview `here <https://en.wikipedia.org/wiki/Comparison_of_agent-based_modeling_software>`_).
The three main distinguishing features of agentpy are the following:

- Agentpy integrates the multiple tasks of agent-based modeling
  - model design, interactive simulations,
  numerical experiments, and data analysis - within a single environment
  and is optimized for interactive computing with IPython and Jupyter.
- Agentpy is designed for scientific use with experiments over multiple runs.
  It provides tools for parameter sampling (similar to NetLogo's BehaviorSpace),
  Monte Carlo experiments, stochastic processes, parallel computing,
  and sensitivity analysis.
- Agentpy is written in Python, one of the world’s most popular
  programming languages that offers a vast number of tools and libraries for scientific use.
  It is further designed for compatibility with established packages like
  numpy, scipy, networkx, pandas, ema_workbench, seaborn, and SALib.

The main alternative to agentpy in Python is `Mesa <https://mesa.readthedocs.io/>`__.
To allow for an comparison of the syntax,
here are two examples for a simple model of wealth transfer,
both of which realize exactly the same operations.
More information on the two models can be found in the documentation
of each framework (:doc:`Agentpy <agentpy_wealth_transfer>` &
`Mesa <https://mesa.readthedocs.io/en/stable/tutorials/intro_tutorial.html#tutorial-description>`_).

+--------------------------------------------+----------------------------------------------+
|**Agentpy**                                 |**Mesa**                                      |
+--------------------------------------------+----------------------------------------------+
|                                            |                                              |
|.. literalinclude:: agentpy_demo.py         |.. literalinclude:: mesa_demo.py              |
|                                            |                                              |
+--------------------------------------------+----------------------------------------------+

The following table further provides a comparison of the main features of each framework.

==========================  ===================================  ======================================
**Feature**                 **Agentpy**                          **Mesa**
| Containers                | Sequence classes                   | Scheduler classes for
                            | like AgentList and AgentDList      | different activation orders
| Topologies                | Spatial grid, continuous space,    | Spatial grid, continuous space,
                            | network                            | network
| Data recording            | Recording methods for variables    | DataCollector class that can
                            | of agents, environments, and       | collect variables of agents
                            | model; as well as reporters        | and model
| Parameter sampling        | Classes for sample generation
                            | and different types of
                            | parameter ranges
| Multi-run experiments     | Experiment class that supports     | BatchRunner class that supports
                            | multiple iterations, parameter     | multiple iterations and parameter
                            | samples, randomization,            | ranges
                            | and parallel processing
| Output data               | DataDict class to store, save,     | Methods to generate dataframes
                            | load, and re-arrange output data   |
| Visualization             | Gridplots, animations,             | Plots and interactive visualization
                            | and interactive visualization      | in a separate web-server
                            | within Jupyter Notebooks
| Analysis                  | Tools for data arrangement and
                            | sensitivity analysis
==========================  ===================================  ======================================