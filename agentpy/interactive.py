"""

Agentpy 
Interactive Module

Copyright (c) 2020 Joël Foramitti

"""

import ipywidgets as widgets 

def interactive(exp,output_function):

        """ 
        
        Returns 'output_function' as an interactive plot 
        Only works in Jupyter Notebook or JupyterLab
        
        More infos at https://ipywidgets.readthedocs.io/en/stable/examples/Output%20Widget.html
        
        """

        model,fixed_parameters,variable_parameters = exp.settings['model'],exp.parameters,exp.settings['param_ranges']
        
        widget_dict = {}

        def var_run(model=model,fixed_parameters=fixed_parameters,**variable_parameters):

            parameters = {}
            parameters.update(fixed_parameters)

            for var_key,value in variable_parameters.items():
                parameters[var_key] = value

            temp_model = model( parameters )
            results = temp_model.run()
            
            output_function(temp_model) 

        for var_key, var_range in variable_parameters.items():

            if type(var_range) == float: # Transform to tupel
                default = fixed_parameters[var_key]
                var_range = [ default * ( 1 - var_range ) , default * ( 1 + var_range ) ]

            widget_dict[var_key] = widgets.FloatSlider(
                description=var_key, 
                value = fixed_parameters[var_key] , 
                min = var_range[0], 
                max = var_range[1], 
                step = (var_range[1] - var_range[0]) / 10 , 
                style = dict(description_width='initial') , 
                layout = {'width': '500px'} )

        out = widgets.interactive_output(var_run, widget_dict)

        return widgets.HBox([ out, widgets.VBox(list(widget_dict.values())) ])
    