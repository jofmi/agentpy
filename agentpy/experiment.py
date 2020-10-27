"""

Agentpy 
Experiment Module

Copyright (c) 2020 Joël Foramitti

"""



import pandas as pd
import networkx as nx
import warnings

from datetime import datetime, timedelta

from .tools import attr_dict, make_list, AgentpyError
from .output import data_dict

class experiment():
    
    """ Experiment for an agent-based model.
    Allows for multiple iterations, parameter samples, and distict scenarios.
    
    Arguments:
        model(class): The model that the experiment should use.
        parameters(dict or list of dict): Parameter dictionary or parameter sample (list of parameter dictionaries).
        name(str,optional): Name of the experiment. Takes model name at default.
        scenarios(str or list,optional): Scenarios that should be tested, if any.
        iterations(int,optional): How often to repeat the experiment (default 1).
        record(bool,optional): Whether to record dynamic variables (default False).
        
    Attributes:
        output(data_dict): Recorded experiment data
    
    """
    
    def __init__( self, 
                 model, 
                 parameters = None, 
                 name = None,
                 scenarios = None,
                 iterations = 1,
                 record = False
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
        self.record = record
        
        # Log
        self.output.log = {}
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
        
        # Document parameters (seperately for fixed & variable)
        df = pd.DataFrame(parameter_sample)
        df.index.rename('sample_id', inplace=True)
        fixed_pars = {}
        
        for col in df.columns:
            s = df[col]
            if len(s.unique()) == 1:
                fixed_pars[s.name] = df[col][0]
                df.drop(col, inplace=True, axis=1)
        
        if fixed_pars and df.empty:
            self.output['parameters'] = fixed_pars
        elif not fixed_pars and not df.empty:
            self.output['parameters'] = df
        else:
            self.output['parameters'] = data_dict({
                'fixed': fixed_pars,
                'varied': df
            })

        ## START EXPERIMENT ##
        
        if display: print( f"Scheduled runs: {n_runs}" )
        t0 = datetime.now() # Time-Stamp Start
        
        combined_output = {}
        
        for i, parameters in enumerate(runs):
            
            for scenario in scenarios:
                
                # Run model for current parameters & scenario
                single_output = self.model(parameters, run_id = i, scenario = scenario).run(display = False)
                
                # Append results to experiment output
                for key,value in single_output.items():
                    
                    # Skip parameters & log
                    if key in ['parameters','log']:
                        continue
                    
                    # Handle variables
                    if self.record and key == 'variables' and isinstance(value,data_dict): 
                        
                        if key not in combined_output: 
                            combined_output[key] = {}
                        
                        for var_key,value in single_output[key].items():
                            
                            if var_key not in combined_output[key]: 
                                combined_output[key][var_key] = []
                            
                            combined_output[key][var_key].append(value)
                    
                    # Handle other output types
                    else: 
                        if key not in combined_output: 
                            combined_output[key] = []
                        combined_output[key].append(value)     
                    
            if display: 
                td = ( datetime.now() - t0 ).total_seconds()
                te = timedelta(seconds= int( td / (i+1) * ( n_runs - i - 1 ) ) )
                print( f"\rCompleted: {i+1}, estimated time remaining: {te}" , end='' ) 
        
        # Combine dataframes
        for key,values in combined_output.items():
            if values and all([isinstance(value,pd.DataFrame) for value in values]):
                self.output[key] = pd.concat(values) 
            elif isinstance(values,dict):
                self.output[key] = data_dict()
                for sk,sv in values.items():
                    self.output[key][sk] = pd.concat(sv) 
            elif key != 'log':
                self.output[key] = values
                
        self.output.log['run_time'] = ct = str( datetime.now() - t0 )
        
        if display: print(f"\nRun time: {ct}\nSimulation finished")
        
        return self.output
