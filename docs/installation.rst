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

Agentpy supports Python 3.6, 3.7, 3.8, and 3.9.
The installation includes the following packages:

- `numpy <https://numpy.org>`_, for scientific computing
- `matplotlib <https://matplotlib.org/>`_, for visualization
- `pandas <https://pandas.pydata.org>`_, for output dataframes
- `networkx <https://networkx.org/documentation/>`_, for network analysis
- `IPython <https://ipython.org/>`_ and `ipywidgets <https://ipywidgets.readthedocs.io/>`_, for interactive computing
- `SALib <https://salib.readthedocs.io/>`_, for sensitivity analysis

These optional packages can further be useful in combination with agentpy,
and are required in some of the tutorials:

- `jupyter <https://jupyter.org/>`_, for interactive computing
- `seaborn <https://seaborn.pydata.org/>`_, for statistical data visualization

Development
-----------

The most recent version of agentpy can be cloned from Github:

.. code-block:: console

	$ git clone https://github.com/JoelForamitti/agentpy.git

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install -e

To include all necessary packages for development, you can use:

.. code-block:: console

    $ pip install -e .['dev']

.. _Github repository: https://github.com/JoelForamitti/agentpy
