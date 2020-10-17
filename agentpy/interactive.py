"""

Agentpy 
Interactive Module

Copyright (c) 2020 Joël Foramitti

"""

import ipywidgets as widgets 
import matplotlib.pyplot as plt

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

        for ax in axs:
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

    
