.. currentmodule:: agentpy

========================================
AgentPy - Agent-based modeling in Python
========================================

.. image:: https://img.shields.io/pypi/v/agentpy.svg
    :target: https://pypi.org/project/agentpy/
.. image:: https://img.shields.io/github/license/joelforamitti/agentpy
    :target: https://github.com/JoelForamitti/agentpy/blob/master/LICENSE
.. image:: https://readthedocs.org/projects/agentpy/badge/?version=latest
    :target: https://agentpy.readthedocs.io/en/latest/?badge=latest
.. image:: https://joss.theoj.org/papers/10.21105/joss.03065/status.svg
    :target: https://doi.org/10.21105/joss.03065

.. raw:: latex

    \chapter{Introduction}

AgentPy is an open-source library for the development and analysis of agent-based models in Python.
The framework integrates the tasks of model design, interactive simulations, numerical experiments,
and data analysis within a single environment. The package is optimized for interactive computing
with `IPython <http://ipython.org/>`_ and `Jupyter <https://jupyter.org/>`_.

**Note:** AgentPy is no longer under active development. For new projects, we recommend using [MESA](https://mesa.readthedocs.io/stable/).

.. rubric:: Quick orientation

- To get started, please take a look at :doc:`installation` and :doc:`overview`.
- For a simple demonstration, check out the :doc:`agentpy_wealth_transfer` tutorial in the :doc:`model_library`.
- For a detailled description of all classes and functions, refer to :doc:`reference`.
- To learn how agentpy compares with other frameworks, take a look at :doc:`comparison`.
- If you are interested to contribute to the library, see :doc:`contributing`.

.. rubric:: Citation

Please cite this software as follows:

.. code-block:: text

    Foramitti, J., (2021). AgentPy: A package for agent-based modeling in Python.
    Journal of Open Source Software, 6(62), 3065, https://doi.org/10.21105/joss.03065

.. only:: html

    .. rubric:: Table of contents

.. toctree::
   :maxdepth: 2

   installation
   overview
   guide
   model_library
   reference
   changelog
   contributing
   about

.. only:: html

    .. rubric:: Indices and tables

    * :ref:`genindex`
    * :ref:`search`
