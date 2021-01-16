---
title: 'Agentpy: A Python package for agent-based modeling'
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

Agentpy is an open-source library for the development and analysis of agent-based models (ABMs) in Python. 
ABMs are computer simulations
to generate and understand emergent dynamics of complex systems 
based on the autonomous behavior of heterogeneous agents.
This method is being increasingly applied in fields like
ecology [@DeAngelis2019], cognitive sciences [@Madsen2019], management [@North2007], 
policy analysis [@Castro2020], economics [@Farmer2009], and sociology [@Bianchi2015].

The aim of agentpy is to provide an intuitive syntax for the creation of models
together with advanced tools for scientific applications.
The framework integrates the tasks of model design, numerical experiments, 
and subsequent data analysis and visualization within a single environment, and is
optimized for interactive computing with [IPython](http://ipython.org/) and [Jupyter](https://jupyter.org/) (see \autoref{fig:example}). A reference of all features
as well as a model library with tutorials and examples can be found in the [documentation](https://agentpy.readthedocs.io/).

![A screenshot of Jupyter Lab with two interactive tutorials from the agentpy model library.\label{fig:example}](docs/agentpy_example.png){ width=80% }

# Overview

The framework follows a nested structure that is illustrated in \autoref{fig:structure}.
The basic building blocks are the agents, which can be designed with custom variables and methods.
These agents can then be placed within different environments like a network or a spatial grid.
A model is used to initiate these objects, perform a simulation, and record data. 
Experiments, in turn, can take a model and run it over multiple iterations and parameter combinations.
The resulting output data can be saved, loaded, and re-arranged for analysis and visualization.

![Nested structure of the agentpy framework.\label{fig:structure}](docs/structure.png){ width=80% }

An example of how to design and execute a simple model can be seen on the left panel of \autoref{fig:example}. 
The syntax of the framework is designed to be both intuitive and efficient. 
For example, to call a method *action* for a whole group of *agents*, one would simply call: `agents.action()`. To select agents with a variable *x* above zero: `agents.select(agents.x > 0)`. To increase this variable by one: `agents.x += 1`. And to record it: `agents.record('x')`. 

The package further aims to provide advanced tools for experimentation and analysis. At the time of writing, these include Monte-Carlo simulation, scenario comparison, parameter sampling, sensitivity analysis, parallel computing, interactive sliders, data manipulation, animations, and plots.
These features make use of – and are compatible with – established Python libraries for scientific computing, including [pandas](https://pandas.pydata.org/), [networkx](https://networkx.org/), [SALib](https://salib.readthedocs.io/), and [seaborn](https://seaborn.pydata.org/).

# Comparison

There are numerous modeling and simulation tools for ABMs [@Abar2017],
each with their own particular focus and style. 
The main alternative to agentpy is [Mesa](https://mesa.readthedocs.io/), 
which aims to be a "Python 3-based counterpart to NetLogo, Repast, or MASON" [@Mesa2016]. 
All of these frameworks traditionally focus on spatial environments and live visual interfaces,
where users can observe dynamics and adjust parameters while the model is running.

Agentpy, in contrast, is more focused on experiments over multiple runs, 
with tools to generate and analyze output data from these experiments. 
It further differs from existing frameworks in both style and structure,
most notably through its nested structure (\autoref{fig:structure}), simple syntax for interactive computing, and direct integration of analysis tools within the same framework.
Further comparison of code examples and features can be found [here](https://agentpy.readthedocs.io/en/latest/comparison.html).

# Acknowledgements

This study has received funding through an ERC Advanced Grant from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement n° 741087).

# References