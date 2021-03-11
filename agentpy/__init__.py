"""
Agentpy - Agent-based modeling in Python
Copyright (c) 2020 Joël Foramitti

Documentation: https://agentpy.readthedocs.io/
Examples: https://agentpy.readthedocs.io/en/latest/model_library.html
Source: https://github.com/JoelForamitti/agentpy
"""

__all__ = [
    '__version__',
    'Model', 'Environment', 'Network', 'Grid', 'Space',
    'Agent', 'AgentList', 'EnvList', 'ObjList', 'AttrList',
    'Experiment',
    'DataDict', 'load',
    'sample', 'sample_discrete', 'sample_saltelli',
    'sensitivity_sobol', 'gridplot', 'animate',
    'AttrDict'
]

# Version
try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python
    import importlib_metadata as metadata  # noqa
__version__ = metadata.version('agentpy')

# Objects
from .lists import AttrList, ObjList, AgentList, EnvList
from .objects import Agent, Environment
from .network import Network
from .grid import Grid
from .space import Space
from .model import Model
from .experiment import Experiment
from .output import DataDict, load
from .sampling import sample, sample_discrete, sample_saltelli
from .analysis import sensitivity_sobol, gridplot, animate
from .tools import AttrDict
