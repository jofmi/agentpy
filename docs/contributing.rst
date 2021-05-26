.. currentmodule:: agentpy

==========
Contribute
==========

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given. You can contribute in many ways:

Types of contributions
----------------------

Report bugs
~~~~~~~~~~~

Report bugs at https://github.com/JoelForamitti/agentpy/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues and `discussion forum <https://github.com/JoelForamitti/agentpy/discussions>`_ for features.
Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

Write documentation
~~~~~~~~~~~~~~~~~~~

Agentpy could always use more documentation, whether as part of the
official agentpy docs, in docstrings, or even on the web in blog posts,
articles, and such. Contributions of clear and simple demonstration models for the :doc:`model_library`
that illustrate a particular application are also very welcome.

Submit feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to write in the agentpy discussion forum at https://github.com/JoelForamitti/agentpy/discussions.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

How to contribute
-----------------

Ready to contribute? Here's how to set up `agentpy` for local development.

1. Fork the `agentpy` repository on GitHub: https://github.com/JoelForamitti/agentpy
2. Clone your fork locally:

.. code-block:: console

    $ git clone git@github.com:your_name_here/agentpy.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

.. code-block:: console

    $ mkvirtualenv agentpy
    $ cd agentpy/
    $ pip install -e .['dev']

4. Create a branch for local development:

.. code-block:: console

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests
   and that the new features are covered by the tests:

.. code-block:: console

    $ coverage run -m pytest
    $ coverage report

6. Commit your changes and push your branch to GitHub:

.. code-block:: console

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull request guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
   For more information, check out the tests directory and https://docs.pytest.org/.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to docs/changelog.rst.
3. The pull request should pass the automatic tests on travis-ci. Check
   https://travis-ci.com/JoelForamitti/agentpy/pull_requests
   and make sure that the tests pass for all supported Python versions.
