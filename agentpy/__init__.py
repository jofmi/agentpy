"""
Agentpy - Agent-based modeling in Python
Copyright (c) 2020 Joël Foramitti

Documentation: https://agentpy.readthedocs.io/
Examples: https://agentpy.readthedocs.io/en/latest/model_library.html
Source: https://github.com/JoelForamitti/agentpy
"""

__all__ = [
    '__version__',
    'Model',
    'Agent', 'AgentList', 'AgentGroup', 'AgentSet',
    'AgentIter', 'AgentGroupIter', 'AttrIter',
    'Grid', 'GridIter', 'Space', 'Network', 'AgentNode',
    'Experiment',
    'DataDict', 'load',
    'Sample', 'Values', 'Range',
    'gridplot', 'animate',
    'AttrDict'
]

from .version import __version__

from .model import Model
from .agent import Agent, MultiAgent
from .sequences import AgentList, AgentGroup, AgentSet
from .sequences import AgentIter, AgentGroupIter, AttrIter
from .network import Network, AgentNode
from .grid import Grid, GridIter
from .space import Space
from .experiment import Experiment
from .datadict import DataDict, load
from .sample import Sample, Values, Range
from .analysis import gridplot, animate
from .tools import AttrDict
