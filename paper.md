---
title: 'Agentpy: A package for agent-based modeling in Python'
tags:
  - Agent-based modeling
  - Complex systems
  - Networks
  - Interactive computing
  - Python
authors:
  - name: Joël Foramitti
    orcid: 0000-0002-4828-7288
    affiliation: "1, 2"
affiliations:
 - name: Institute of Environmental Science and Technology, Universitat Autònoma de Barcelona, Spain
   index: 1
 - name: Institute for Environmental Studies, Vrije Universiteit Amsterdam, The Netherlands
   index: 2
date: 16.01.2020
bibliography: paper.bib
---

# Introduction

Agent-based models are computer simulations based on the autonomous behavior of heterogeneous agents. They are used to generate and understand the emergent dynamics of complex systems, with applications in fields like ecology [@DeAngelis2019], cognitive sciences [@Madsen2019], management [@North2007], policy analysis [@Castro2020], economics [@Farmer2009], and sociology [@Bianchi2015].

Agentpy is an open-source library for the development and analysis of agent-based models. It aims to provide a new intuitive syntax for the creation of models together with advanced tools for scientific applications. The framework is written in Python 3, and optimized for interactive computing with [IPython](http://ipython.org/) and [Jupyter](https://jupyter.org/). A reference of all features as well as a model library with tutorials and examples can be found in the documentation (https://agentpy.readthedocs.io/).

# Statement of Need

There are numerous modeling and simulation tools for agent-based models, each with their own particular focus and style [@Abar2017]. Most notable examples are NetLogo [@Netlogo], which is written in Scala/Java and has become the most established tool in the field; and Mesa [@Mesa2016], a more recent framework that has popularized the development of agent-based models in Python. 

Agentpy's main feature in comparison to these existing tools is that it integrates the many different tasks of agent-based modeling within a single environment. These tasks include the creation of custom agent, environment, and model types; interactive simulations (similar to the traditional NetLogo interface); numeric experiments over multiple runs; and subsequent data analysis of the output.

The software is further designed for scientific applications, and includes tools for parameter sampling (similar to NetLogo's BehaviorSpace), Monte Carlo experiments, random number generation, parallel computing, and sensitivity analysis; as well as compatibility with established Python packages like [EMA Workbench](https://emaworkbench.readthedocs.io/), [NetworkX](https://networkx.org/), [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [SALib](https://salib.readthedocs.io/), [SciPy](https://www.scipy.org/),  and [seaborn](https://seaborn.pydata.org/).

# Basic structure

The agentpy framework follows a nested structure that is illustrated in \autoref{fig:structure}. The basic building blocks are the agents, which can be placed within (multiple) environments with different topologies like a network, a spatial grid, or a continuous space. Models are used to initiate these objects, perform a simulation, and record data. Experiments can run a model over multiple iterations and parameter combinations. The resulting output data can then be saved and re-arranged for analysis and visualization.

![Nested structure of the agentpy framework.\label{fig:structure}](docs/graphics/structure.png){ width=70% }

# Acknowledgements

This study has received funding through an ERC Advanced Grant from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement n° 741087).

# References