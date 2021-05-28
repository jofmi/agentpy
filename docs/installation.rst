.. currentmodule:: agentpy
.. highlight:: shell

============
Installation
============

To install the latest release of agentpy,
run the following command on your console:

.. code-block:: console

	$ pip install agentpy

Dependencies
------------

Agentpy supports Python 3.6 and higher.
The installation includes the following packages:

- `numpy <https://numpy.org>`_ and `scipy <https://docs.scipy.org/>`_, for scientific computing
- `matplotlib <https://matplotlib.org/>`_, for visualization
- `pandas <https://pandas.pydata.org>`_, for data manipulation
- `networkx <https://networkx.org/documentation/>`_, for networks/graphs
- `SALib <https://salib.readthedocs.io/>`_, for sensitivity analysis

These optional packages can further be useful in combination with agentpy:

- `jupyter <https://jupyter.org/>`_, for interactive computing
- `ipysimulate <https://ipysimulate.readthedocs.io/>`_ >= 0.2.0, for interactive simulations
- `ema_workbench <https://emaworkbench.readthedocs.io/>`_, for exploratory modeling
- `seaborn <https://seaborn.pydata.org/>`_, for statistical data visualization

Development
-----------

The most recent version of agentpy can be cloned from Github:

.. code-block:: console

	$ git clone https://github.com/JoelForamitti/agentpy.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -e

To include all necessary packages for development & testing, you can use:

.. code-block:: console

    $ pip install -e .['dev']

.. _Github repository: https://github.com/JoelForamitti/agentpy
