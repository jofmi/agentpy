"""
Agentpy - Agent-based modeling in Python
Copyright (c) 2020 Joël Foramitti

Documentation: https://agentpy.readthedocs.io/
Examples: https://agentpy.readthedocs.io/en/latest/model_library.html
Source: https://github.com/JoelForamitti/agentpy
"""

__all__ = [
    '__version__',
    'Model', 'Environment', 'Network', 'Grid',
    'Agent', 'AgentList', 'EnvList', 'ObjList', 'AttrList',
    'Experiment',
    'DataDict', 'load',
    'sample', 'sample_discrete', 'sample_saltelli',
    'sensitivity_sobol', 'gridplot', 'animate',
    'AttrDict'
]
# Meta-data
__version__ = "0.0.6.dev"

# Objects
from .framework import Model, Environment, Network, Grid
from .framework import Agent, AgentList, EnvList, ObjList, AttrList
from .experiment import Experiment
from .output import DataDict, load
from .sampling import sample, sample_discrete, sample_saltelli
from .analysis import sensitivity_sobol, gridplot, animate
from .tools import AttrDict
