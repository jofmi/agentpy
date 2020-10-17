"""

Agentpy 
Experiment Module

Copyright (c) 2020 Joël Foramitti

"""


import numpy as np
import pandas as pd
import networkx as nx
import warnings

from datetime import datetime, timedelta

from .tools import attr_dict, make_list, AgentpyError
from .output import data_dict

import itertools
from SALib.sample import saltelli as SALibSaltelli


class experiment():
    
    """ Experiment for an agent-based model.
    Allows for multiple iterations, parameter samples, and distict scenarios.
    
    Arguments:
        model(class): The model that the experiment should use.
        parameters(dict or list): Parameters or parameter sample.
        name(str,optional): Name of the experiment. Takes model name at default.
        scenarios(str or list,optional): Scenarios that should be tested, if any.
        iterations(int,optional): How often to repeat the experiment (default 1).
        output_vars(bool,optional): Whether to record dynamic variables. Default False.
        
    Attributes:
        output(data_dict): Recorded experiment data
    
    """
    
    def __init__( self, 
                 model, 
                 parameters = None, 
                 name = None,
                 scenarios = None,
                 iterations = 1,
                 output_vars = False
                ):
        
        # Experiment objects
        self.model = model
        self.parameters = parameters 
        self.output = data_dict()
        
        # Experiment settings 
        if name: self.name = name
        elif model: self.name = model.__name__
        else: self.name = 'experiment'
        self.scenarios = scenarios
        self.iterations = iterations
        self.output_vars = output_vars
        
        # Log
        self.output.log['name'] = self.name
        self.output.log['time_stamp'] = str(datetime.now())
        self.output.log['iterations'] = iterations
        if scenarios: self.output.log['scenarios'] = scenarios
    
    def run(self, display=True):
             
        """ Executes the simulation of the experiment. 
        It will run the model once for each set of parameters, 
        and will repeat this process for each iteration.
        
        Arguments:
            display(bool,optional): Whether to display simulation progress (default True). 
            
        Returns:
            data_dict: Recorded experiment data, also stored in `experiment.output`.
        """
        parameter_sample = make_list(self.parameters,keep_none=True)
        scenarios = make_list(self.scenarios,keep_none=True)
        runs = parameter_sample * self.iterations
        self.output.log['n_runs'] = n_runs = len(runs)
        
        self.output.parameter_sample = pd.DataFrame(parameter_sample)
        self.output.parameter_sample.index.rename('sample_id', inplace=True)

        if display: print( f"Scheduled runs: {n_runs}" )
        t0 = datetime.now() # Time-Stamp Start
        
        temp_output = {}
        
        for i, parameters in enumerate(runs):
            
            for scenario in scenarios:
                
                # Run model for current parameters & scenario
                output = self.model(parameters, run_id = i, scenario = scenario).run(display=False)
                
                # Append results to experiment output
                for key,value in output.items():
                    
                    # Skip vars?
                    if not self.output_vars and 'vars' in key: 
                        continue
                    
                    # Initiate new key
                    if key not in temp_output: 
                        temp_output[key] = []
                    
                    # Append output
                    temp_output[key].append(value)     
                    
            if display: 
                td = ( datetime.now() - t0 ).total_seconds()
                te = timedelta(seconds= int( td / (i+1) * ( n_runs - i - 1 ) ) )
                print( f"\rCompleted: {i+1}, estimated time remaining: {te}" , end='' ) 
        
        # Combine dataframes
        for key,values in temp_output.items():
            if values and all([isinstance(value,pd.DataFrame) for value in values]):
                self.output[key] = pd.concat(values) 
            elif key != 'log':
                self.output[key] = values
                
        self.output.log['run_time'] = ct = str( datetime.now() - t0 )
        
        if display: print(f"\nRun time: {ct}\nSimulation finished")
        
        return self.output




def create_sample_discrete(parameter_ranges):

    """ Creates a parameter_sample out of all possible combinations given in parameter tuples """
    
    def make_tuple(v):
        if isinstance(v,tuple): return v
        else: return (v,)
    
    param_ranges_values = [ make_tuple(v) for k,v in parameter_ranges.items() ]
    parameter_combinations = list(itertools.product(*param_ranges_values))
    parameter_sample = [ { k:v for k,v in zip(parameter_ranges.keys(),parameters) } for parameters in parameter_combinations ]
    
    return parameter_sample


def saltelli(param_ranges,N,**kwargs):    

    """ Creates saltelli parameter sample with the SALib Package (https://salib.readthedocs.io/) """

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


def sample(param_ranges,mode='discrete',**kwargs):

    """
    Returns parameter sample (list of dict)
    
    Arguments:
        param_ranges(dict)
        mode(str): Sampling method. Options are:
            'discrete' - tuples are given of style (value1,value2,...); 
            'saltelli' - tuples are given of style (min_value,max_value)
    """
    
    if mode == 'discrete': parameters = create_sample_discrete(param_ranges,**kwargs)
    elif mode == 'saltelli': parameters = saltelli(param_ranges,**kwargs)
    else: raise ValueError(f"mode '{mode}' does not exist.")
        
    return parameters