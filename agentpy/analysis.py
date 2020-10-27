"""

Agentpy
Analysis Module

Copyright (c) 2020 Joël Foramitti

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.interpolate import griddata
from SALib.analyze import sobol

from .tools import make_list
from .framework import agent_list
     
        
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
        df['parameter'] = output.parameters.varied.keys()
        df['measure'] = measure
        df = df.set_index(['measure','parameter'])
        dfs_SI.append(df)

        keys1 = ['S1_conf','ST_conf']
        s3 = {k:v for k,v in SI.items() if k in keys1}
        df = pd.DataFrame(s3)
        df['parameter'] = output.parameters.varied.keys()
        df['measure'] = measure
        df = df.set_index(['measure','parameter'])
        df.columns = ['S1','ST']
        dfs_SI_conf.append(df)
    
    output['sensitivity'] = pd.concat(dfs_SI)
    output['sensitivity_conf'] = pd.concat(dfs_SI_conf) 
    
    # (!) Second-Order Entries Missing
    
    return output['sensitivity']




import ipywidgets as widgets 
import matplotlib.pyplot as plt

from .tools import make_list
from matplotlib import animation 

def animate(model, parameters, fig, axs, plot, skip_t0 = False, **kwargs):
        
    """ Returns an animation of the model simulation """
    
    m = model(parameters)
    m._stop = False
    m.setup()
    m.update()

    step0 = True
    step00 = not skip_t0

    def frames():

        nonlocal m, step0, step00

        while not m.stop_if() and not m._stop:

            if step0: step0 = False
            elif step00: step00 = False
            else:
                m.t += 1
                m.step() 
            m.update()
            m.create_output()
            yield m.t 

    def update(t, m, axs):

        for ax in make_list(axs):
            ax.clear()

        if m.t == 0 and skip_t0: pass
        else: plot( m, axs )
    
    ani = animation.FuncAnimation(fig, update, frames=frames, fargs=(m, axs), **kwargs)
    plt.close() # Don't display static plot
    
    return ani


def interactive(model,param_ranges,output_function,*args,**kwargs):

        """ 
        
        Returns 'output_function' as an interactive ipywidget
        More infos at https://ipywidgets.readthedocs.io/
        
        """
             
        def make_param(param_updates):
            
            parameters = dict(param_ranges) # Copy
            i = 0 
            
            for key, value in parameters.items():
                
                if isinstance(v,tuple): 
                    parameters[key] = param_updates[i]
                    i += 1
      
        def var_run(**param_updates):
            
            parameters = dict(param_ranges)
            parameters.update(param_updates)#make_param(param_updates)
            temp_model = model(parameters)
            temp_model.run(display=False)
            
            output_function(temp_model.output,*args,**kwargs) 
 
        # Create widget dict
        widget_dict = {}
        param_ranges_tuples = { k:v for k,v in param_ranges.items() if isinstance(v,tuple) }
        for var_key, var_range in param_ranges_tuples.items():

            widget_dict[var_key] = widgets.FloatSlider(
                description=var_key, 
                value = (var_range[1] - var_range[0]) / 2 , 
                min = var_range[0], 
                max = var_range[1], 
                step = (var_range[1] - var_range[0]) / 10 , 
                style = dict(description_width='initial') , 
                layout = {'width': '300px'} )

        out = widgets.interactive_output(var_run, widget_dict)

        return widgets.HBox([ widgets.VBox(list(widget_dict.values())), out ])

    


    
def gridplot(model,ax,grid_key,attr_key,color_dict):
    
    def apply_colors(grid,color_assignment,final_type,attr_key):
        if not isinstance(grid[0],final_type):
            return [apply_colors(subgrid,color_assignment,final_type,attr_key) for subgrid in grid]
        else:
            return [colors.to_rgb(color_assignment(i,attr_key)) for i in grid]
    
    def color_assignment(a_list,attr_key):

        if len(a_list) == 0: return color_dict['empty']
        else: return color_dict[a_list[0][attr_key]]
    
    grid = model.envs[grid_key].grid
    color_grid = apply_colors(grid,color_assignment,agent_list,attr_key)
    im = ax.imshow(color_grid)   
    
    

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









