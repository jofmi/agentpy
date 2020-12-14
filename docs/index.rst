.. currentmodule:: agentpy

========================================
Agentpy - Agent-based modeling in Python
========================================

.. raw:: latex

    \chapter{Introduction}

Agentpy is an open-source framework for the development and analysis of
agent-based models in Python.
To get started, please take a look at :doc:`installation` and :doc:`overview`.
For a simple demonstration, check out the :doc:`agentpy_wealth_transfer` model.
Further demonstration models can be found in the :doc:`model_library`.
For a detailled description of all classes and functions, refer to :doc:`reference`.

This project is still in an early stage of development.
If you have feedback, need help, or want to contribute,
please write to joel.foramitti@uab.cat.

.. rubric:: Main features

*Aim 1:* Intelligent syntax for complex models

- Custom agent, environment, and network types
- Easy selection and manipulation of agent groups
- Support of multiple environments for interaction

*Aim 2:* Advanced tools for scientific applications

- :ref:`Experiments<overview_experiments>` with repeated iterations and parallel processing
- Parameter sampling and scenario comparison
- Output data that can be saved, loaded, and re-arranged
- Sensitivity analysis and (animated) visualizations

*Aim 3:* Compatibility with established Python libraries

- Data analysis with pandas and SALib
- Statistical visualization with seaborn
- Networks and graphs with networkx
- Interactive output with ipywidgets

.. only:: html

    .. rubric:: Table of contents

.. toctree::
   :maxdepth: 2

   installation
   overview
   model_library
   reference
   changelog

.. only:: html

    .. rubric:: Indices and tables
    
    * :ref:`genindex`
    * :ref:`search`