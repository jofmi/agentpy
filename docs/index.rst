.. currentmodule:: agentpy

========================================
Agentpy - Agent-based modeling in Python
========================================

.. image:: https://img.shields.io/pypi/v/agentpy.svg
    :target: https://pypi.org/project/agentpy/
.. image:: https://img.shields.io/github/license/joelforamitti/agentpy
    :target: https://github.com/JoelForamitti/agentpy/blob/master/LICENSE
.. image:: https://travis-ci.com/JoelForamitti/agentpy.svg?branch=master
    :target: https://travis-ci.com/JoelForamitti/agentpy
.. image:: https://codecov.io/gh/JoelForamitti/agentpy/branch/master/graph/badge.svg?token=NTW99HNGB0
    :target: https://codecov.io/gh/JoelForamitti/agentpy


.. raw:: latex

    \chapter{Introduction}

Agentpy is an open-source library for the development and analysis of agent-based models in Python.
The framework integrates the tasks of model design, numerical experiments,
and data analysis within a single environment, and is optimized for interactive computing
with `IPython <http://ipython.org/>`_ and `Jupyter <https://jupyter.org/>`_.
If you have questions or ideas for improvements, please visit the
`discussion forum <https://github.com/JoelForamitti/agentpy/discussions>`_
or subscribe to the `agentpy mailing list <https://groups.google.com/g/agentpy>`_.

.. rubric:: Quick orientation

- To get started, please take a look at :doc:`installation` and :doc:`overview`.
- For a simple demonstration, check out the :doc:`agentpy_wealth_transfer` tutorial in the :doc:`model_library`.
- For a detailled description of all classes and functions, refer to :doc:`reference`.
- To learn how agentpy compares with other frameworks, take a look at :doc:`comparison`.

.. rubric:: Example

*A screenshot of Jupyter Lab with two interactive tutorials from the model library:*

.. image:: agentpy_example.png
  :width: 700
  :alt: Screenshot of Jupyter Lab with two interactive tutorials from the model library

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

- Interactive computing with Jupyter/IPython
- Data analysis with pandas and SALib
- Network analysis with networkx
- Visualization with seaborn

.. only:: html

    .. rubric:: Table of contents

.. toctree::
   :maxdepth: 2

   installation
   overview
   comparison
   model_library
   reference
   changelog

.. only:: html

    .. rubric:: Indices and tables
    
    * :ref:`genindex`
    * :ref:`search`