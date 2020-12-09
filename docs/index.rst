.. currentmodule:: agentpy

========================================
Agentpy - Agent-based modeling in Python
========================================

.. raw:: latex

    \chapter{Introduction}

Agentpy is an open-source framework for the development and analysis of
agent-based models in Python.
This project is still in an early stage of development.
If you have feedback, need help, or want to contribute,
please write to joel.foramitti@uab.cat.

To get started, please take a look at :doc:`installation` and :doc:`overview`.
For a simple demonstration, check out the :doc:`agentpy_wealth_transfer` model.
Further demonstration models can be found in the :doc:`model_library`.
For a detailled description of all classes and functions, refer to :doc:`reference`.

.. rubric:: Main features

- Design of agent-based models with complex procedures.
- Creation of custom agent types, environments, and networks.
- Agent lists that can forward attribute calls and select agent groups.
- Experiments with repeated iterations, parameter samples, and distinct scenarios.
- Output data that can be saved, loaded, and re-arranged for further analysis.
- Tools for sensitivity analysis, interactive output, animations, and plots.
- Compatibility with NumPy, pandas, matplotlib, seaborn, networkx, and SALib.

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