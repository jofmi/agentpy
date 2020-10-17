"""

Agentpy
Analysis Module

Copyright (c) 2020 Joël Foramitti

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from SALib.analyze import sobol

from .tools import make_list
     
        
def sensitivity( output, param_ranges, measures = None, **kwargs ): 

    """ 
    Returns Sobol Sensitivity Indices based on the SALib Package and adds them to output
    (https://salib.readthedocs.io/) 
    """

    # STEP 1 - Convert param_ranges to SALib Format 

    param_ranges_tuples = { k:v for k,v in param_ranges.items() if isinstance(v,tuple) }
    
    param_ranges_salib = {    
        'num_vars': len( param_ranges_tuples ),
        'names': list( param_ranges_tuples.keys() ),
        'bounds': []
    }

    for var_key, var_range in param_ranges_tuples.items():

        param_ranges_salib['bounds'].append( [ var_range[0] , var_range[1]  ] )

    # STEP 2 - Calculate Sobol Sensitivity Indices
    
    if measures is None:
        measures = output.measures.columns
        
    if isinstance(measures,str):
        measures = make_list(measures)
    
    dfs_SI = []
    dfs_SI_conf = []
    
    for measure in measures:
    
        Y = np.array( output.measures[measure] )

        SI = sobol.analyze( param_ranges_salib , Y , **kwargs )
        
        # Make dataframes out of sensitivities
        keys = ['S1','ST']
        s2 = {k:v for k,v in SI.items() if k in keys}
        df = pd.DataFrame(s2)
        df['parameter'] = output.ps_keys()
        df['measure'] = measure
        df = df.set_index(['measure','parameter'])
        dfs_SI.append(df)

        keys1 = ['S1_conf','ST_conf']
        s3 = {k:v for k,v in SI.items() if k in keys1}
        df = pd.DataFrame(s3)
        df['parameter'] = output.ps_keys()
        df['measure'] = measure
        df = df.set_index(['measure','parameter'])
        df.columns = ['S1','ST']
        dfs_SI_conf.append(df)
    
    output['sensitivity'] = pd.concat(dfs_SI)
    output['sensitivity_conf'] = pd.concat(dfs_SI_conf) 
    
    # (!) Second-Order Entries Missing
    
    return output['sensitivity']


def phaseplot(data,x,y,z,n,fill=True,**kwargs):
    
    """ Creates a contour plot displaying the interpolated 
    sensitivity between the parameters x,y and the measure z """
    
    # Create grid
    x_vals = np.linspace( min(data[x]) , max(data[x]), n)
    y_vals = np.linspace( min(data[y]) , max(data[y]), n)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Interpolate Z
    Z = griddata((data[x],data[y]),data[z],(X,Y))

    # Create contour plot
    if fill: img = plt.contourf(X,Y,Z,**kwargs)
    else: img = plt.contour(X,Y,Z,**kwargs)

    # Create colorbar
    plt.colorbar(mappable=img)

    # Labels
    plt.title(z)
    plt.xlabel(x)
    plt.ylabel(y)
    
    plt.show()









