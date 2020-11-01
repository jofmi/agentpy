"""

Agentpy 
Sampling Module

Copyright (c) 2020 Joël Foramitti

"""

import itertools
import numpy as np

from SALib.sample import saltelli as SALibSaltelli

def sample(parameter_ranges,N):

    """ Creates a parameter_sample out of all possible combinations given in parameter tuples 
    uses np.arange(), tuples are given of style (min_value,max_value)"""
    
    def make_tuple(v):
        if isinstance(v,tuple): return v
        else: return (v,)
    
    for k,v in parameter_ranges.items():
        if isinstance(v,tuple):
            parameter_ranges[k] = np.arange(v[0],v[1],(v[1]-v[0])/N)
        else:
            parameter_ranges[k] = [v]
        
    parameter_combinations = list(itertools.product(*parameter_ranges.values()))
    parameter_sample = [ { k:v for k,v in zip(parameter_ranges.keys(),parameters) } for parameters in parameter_combinations ]
    
    return parameter_sample


def sample_discrete(parameter_ranges):

    """ Creates a parameter_sample out of all possible combinations given in parameter tuples 
    uses SALib.saltelli(), tuples are given of style (min_value,max_value)"""
    
    def make_tuple(v):
        if isinstance(v,tuple): return v
        else: return (v,)
    
    param_ranges_values = [ make_tuple(v) for k,v in parameter_ranges.items() ]
    parameter_combinations = list(itertools.product(*param_ranges_values))
    parameter_sample = [ { k:v for k,v in zip(parameter_ranges.keys(),parameters) } for parameters in parameter_combinations ]
    
    return parameter_sample


def sample_saltelli(param_ranges,N,**kwargs):    

    """ Creates saltelli parameter sample with the SALib Package (https://salib.readthedocs.io/) 
    'discrete' - tuples are given of style (value1,value2,...)  """

    # STEP 1 - Convert param_ranges to SALib Format (https://salib.readthedocs.io/)
    
    param_ranges_tuples = { k:v for k,v in param_ranges.items() if isinstance(v,tuple) }
    
    param_ranges_salib = {    
        'num_vars': len( param_ranges_tuples ),
        'names': list( param_ranges_tuples.keys() ),
        'bounds': []
    }

    for var_key, var_range in param_ranges_tuples.items():

        param_ranges_salib['bounds'].append( [ var_range[0] , var_range[1]  ] )

    # STEP 2 - Create SALib Sample

    salib_sample = SALibSaltelli.sample(param_ranges_salib,N,**kwargs)

    # STEP 3 - Convert back to Agentpy Parameter Dict List

    ap_sample = []

    for param_instance in salib_sample:

        parameters = {}
        parameters.update( param_ranges )

        for i, key in enumerate(param_ranges_tuples.keys()):
            parameters[key] = param_instance[i]

        ap_sample.append( parameters )

    return ap_sample


