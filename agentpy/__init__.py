"""

Agentpy
Initialization Module

Copyright (c) 2020 Joël Foramitti
    
"""

from .framework import model, environment, network, agent, agent_list, env_dict
from .output import data_dict, load, save

from .experiment import experiment, sample

from .interactive import interactive, animate
from .analysis import sensitivity, phaseplot

from .tools import attr_dict, attr_list

# Aliases
exp = experiment
