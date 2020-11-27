"""
Agentpy - Agent-based modeling in Python
Copyright (c) 2020 Joël Foramitti

Documentation: https://agentpy.readthedocs.io/
Source: https://github.com/JoelForamitti/agentpy
Examples: https://agentpy.readthedocs.io/en/latest/model_library.html
"""

# Meta-data
__version__ = "0.0.4"
__all__ = [
    '__version__',
    'Model', 'Environment', 'Network', 'Grid',
    'Agent', 'AgentList', 'EnvDict',
    'Experiment',
    'DataDict', 'load',
    'sample', 'sample_discrete', 'sample_saltelli',
    'sobol_sensitivity', 'gridplot', 'interactive', 'animate'
]

# Objects
from .framework import Model, Environment, Network, Grid
from .framework import Agent, AgentList, EnvDict
from .experiment import Experiment
from .output import DataDict, load
from .sampling import sample, sample_discrete, sample_saltelli
from .analysis import sobol_sensitivity, gridplot, interactive, animate
