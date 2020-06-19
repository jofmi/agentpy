"""

Agentpy
Analysis Module

Copyright (c) 2020 Joël Foramitti

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from scipy import stats
from SALib.analyze import sobol



### Helper Functions ###

def make_list(element):
    
    """ Turns element into a list if it is not of type list or tuple """
    
    if not isinstance(element, (list, tuple)): element = [element]
        
    return element



### Main Functions ###
    

def plot(ax, experiment, var_key, env_key=None, agent_var=False, scenarios=None, agent_type=None, trange= None, xlabel='step'): #, **kwargs):
    
    """ Plots the output of an experiment """
    
    df = experiment.output
    
    # Prepare Output
    
    if trange: df = df[df['t'] in trange]
    else: df = df[df['t']!=0] # Remove round 0
    if env_key: df = df[df['env_key']==env_key]
    
    # Case 1 - Single Run
    
    if len(df['run_id'].unique()) == 1: 
        
        if not agent_var: # Dynamic Environment Variable
            
            df = df[df['var_key']==var_key] # Select var_key
            df = df[df['agent_id'].isnull()] # Select env_vars
            
            if not scenarios: scenarios = df['scenario'].unique()
                
            for scenario in scenarios:
            
                df1 = df[df['scenario']==scenario] # Select scenario
                xy = df1.set_index('t')['value'] # Create time series
                label = var_key + ' ( ' + scenario + ' )'

                ax.plot(xy, label = label) #,**kwargs)
                
                ax.legend()
                
        else: # Dynamic Agent Variable
            
            df = df[df['var_key']==var_key] # Select var_key
            df = df[df['agent_id'].notnull()] # Select agent_vars
            df = df[df['scenario']== scenarios ] # Select scenario
            
            for agent_id in df['agent_id'].unique(): 

                xy = df[df['agent_id']==agent_id] # Select agent
                xy = xy.set_index('t')['value'] # Create time-series

                ax.plot(xy,label=f'Agent {agent_id}') 
                ax.set_ylabel(var_key)
             
        ax.set_xlabel(xlabel)
 

    # Case 2 - Multiple Runs
    
    else: 
        
        if not agent_var:
            
            xy = df[(df['var_key']==var_key) & (df['agent_id']!=np.NaN) ]
            sns.lineplot(x="t", y='value', data=xy, ax=ax, label=var_key)
            
        else: raise ValueError('Plot of agent_vars for multiple runs not supported yet')

            

def heatmap(ax,experiment,x_key,y_key,z_key): 
    
    """ Plots a heatmap that shows how the measure 'z_key' is affected the variation of two parameters 'x_key' & 'y_key'  """
    
    df = experiment.measures
  
    x = [ p[x_key] for p in experiment.param_sample ]  
    y = [ p[y_key] for p in experiment.param_sample ] 
    z = list( df[df['measure_key']==z_key]['value'] )

    data = stats.binned_statistic_2d(x, y, values = z, statistic ='mean', bins = [10, 10])

    sns.heatmap(data[0], cmap="YlOrRd", cbar_kws = {'label':z_key}, ax=ax )
    
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)


    
def sensitivity( experiment, sa_var_key, display = True ): 

    """ Returns Sobol Sensitivity Indices based on the SALib Package (https://salib.readthedocs.io/) """
    
    # STEP 1 - Convert param_ranges to SALib Format 
    
    param_ranges = experiment.settings['param_ranges']
    param_ranges_salib = {    
        'num_vars': len( param_ranges ),
        'names': list( param_ranges.keys() ),
        'bounds': []
    }

    for var_key, var_range in param_ranges.items():

        if isinstance(var_range, (int, float)): # Transform to [min,max] if given as percentage
            default = experiment.settings['parameters'][var_key]
            var_range = [ default * ( 1 - var_range ) , default * ( 1 + var_range ) ]

        param_ranges_salib['bounds'].append( [ var_range[0] , var_range[1]  ] )
    
    # STEP 2 - Calculate Sobol Sensitivity Indices
    
    df = experiment.measures
    df = df[df['measure_key']==sa_var_key] 

    Y=np.array(df['value'])

    Si = sobol.analyze( param_ranges_salib , Y , print_to_console=display , calc_second_order=False)
    
    return Si
    
