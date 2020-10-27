"""

Agentpy
Initialization Module

Copyright (c) 2020 Joël Foramitti
    
"""

from .framework import model, environment, network, grid, agent, agent_list, env_dict

from .sample import sample, sample_discrete, sample_saltelli
from .experiment import experiment

from .output import data_dict, load, save

from .analysis import sensitivity, phaseplot, gridplot, interactive, animate

from .tools import attr_dict, obj_list

# Aliases
exp = experiment
