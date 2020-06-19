"""

Agentpy 
Experiment Module

Copyright (c) 2020 Joël Foramitti

"""


import numpy as np
import warnings 

from pandas import concat
from SALib.sample import saltelli
from time import clock

from os import listdir, makedirs
from pickle import dump,load


class experiment():
    
    """
    
    Experiment Class
    
    Performs experiment based on the passed settings
    The method run() performs the simulation, which generates the dataframes 'output' and 'measures'
    
    """
    
    def __init__( self, settings ):
            
        self.output = None
        self.measures = None
        self.param_sample = None
        
        # Set default settings
        self.settings = {
            'name':'unnamed',
            'scenarios':['default'],
            'multiple_runs': False,
            'sampling_method': 'saltelli',
            'sampling_factor': 1,
            'param_ranges': None,
            'display': True,
            'save': False
        }
        
        # Update passed settings
        self.settings.update( settings )  
        self.parameters = self.settings['parameters']
        
        # Case 1 - Single run
        if not self.settings['multiple_runs']: 
            self.param_sample = [ self.settings['parameters'] ] 
            
        # Case 2 - Multiple runs with fixed parameters
        elif self.settings['param_ranges'] == None: 
            self.param_sample = [ self.settings['parameters'] ] * self.settings['sampling_factor'] 
        
        # Case 3 - Multiple runs with variable parameters  
        else: 
            self.param_sample = self.create_sample()               
               
        self.sample_size = len( self.param_sample )
            
        
        
    def create_sample(self):
        
        """ Creates parameter sample based on passed sampling method ( currently only supports 'saltelli' ) """
        
        param_ranges = self.settings['param_ranges']
                
        # STEP 1 - Convert param_ranges to SALib Format (https://salib.readthedocs.io/)
        
        param_ranges_salib = {    
            'num_vars': len( param_ranges ),
            'names': list( param_ranges.keys() ),
            'bounds': []
        }

        for var_key, var_range in param_ranges.items():

            if isinstance(var_range, (int, float)): # Transform to [min,max] if given as percentage
                default = self.settings['parameters'][var_key]
                var_range = [ default * ( 1 - var_range ) , default * ( 1 + var_range ) ]

            param_ranges_salib['bounds'].append( [ var_range[0] , var_range[1]  ] )
        
        # STEP 2 - Create SALib Sample
        
        if self.settings['sampling_method'] == 'saltelli':
            
            salib_sample = saltelli.sample(param_ranges_salib, self.settings['sampling_factor'], calc_second_order=False)
            
        else: raise ValueError("sampling_method in settings is not supported")
          
        # STEP 3 - Convert back to Agentpy Parameter Dict
        
        ap_sample = []

        for param_instance in salib_sample:

            parameters = {}
            parameters.update( self.settings['parameters'] )

            for i, key in enumerate(param_ranges.keys()):
                parameters[key] = param_instance[i]

            ap_sample.append( parameters )

        return ap_sample
    
    
        
    def run(self):
        
        """ Performs the experiment """
        
        output_list = []
        measure_list = []
        
        model = self.settings['model']
        parameters = self.settings['parameters']
            
        t0 = clock() # Time-Stamp
        td = 0

        if self.settings['display']: print( "Scheduled runs: " , self.sample_size )

        for i, parameters in enumerate( self.param_sample ):

            for scenario in self.settings['scenarios']:

                temp_model = model( parameters, run_id = i, scenario = scenario )
                output_list.append( temp_model.run() ) # (!)
                measure_list.append( temp_model.measures )

            td = clock() - t0
            te = int( td / (i+1) * ( self.sample_size - i - 1 ) )

            if self.settings['multiple_runs'] and self.settings['display']: 
                print( f"\rCompleted: {i+1}, estimated time remaining: {int(te)} seconds" , end='' ) 
                
        if self.settings['display']:
            
            if self.settings['multiple_runs']: print("\n")
            print(f"Simulation complete\nTotal run time: {int(td)} seconds")

        self.model = temp_model # Keep latest model for analysis
        self.measures = concat(measure_list) # Dataframe for evaluation measures
        self.output = concat(output_list) # Creates pandas dataframe

        if self.settings['save']: self.save()
            
            
        
    def save(self):      

        """ Saves the experiment and its results in the subdirectory ap_output """
        
        # Create subdirectory if it doesn't exist
        if 'ap_output' not in listdir(): makedirs('ap_output')
        
        # Identify existing files and create id
        output_files = listdir('ap_output')  
        
        file_name = 'experiment_'+self.settings['name']+'_'
        file_name_len = len(file_name)
        
        old_ids = [ int(l[file_name_len:-4]) for l in output_files if l[:file_name_len]==file_name]
        
        if len(old_ids)>0: new_id = max( old_ids ) + 1
        else: new_id = 0
        
        # Save experiment
        file = open('ap_output/experiment_'+self.settings['name']+f'_{new_id}.obj', 'wb')
        dump( self , file)
        file.close()        
    

    def get_measure(self,var_key,steps=None):

        """ Selects a variable from the experiments' measures """

        df = self.measures
        df = df[ df['var_key'] == var_key ]
        if steps: df = df[ df['t'] in steps ]

        return df
    
    
    def get_var(self,var_key,steps=None,agent_var=False):

        """ Selects a variable from the experiments' output """


        df = self.output
        if agent_var: df = df[ df['agent_id'].notnull() ]
        else: df = df[ df['agent_id'].isnull() ]
        df = df[ df['var_key'] == var_key ]
        if steps: df = df[ df['t'] in steps ]

        return df

        
        
def load_experiment(exp_name='unnamed',exp_id=None):
         
    """ Loads experiment of name 'exp_name' (default 'unnamed') and id 'exp_id' (default None) """  
    
    # Select latest output file
    if exp_id == None: 
        if 'ap_output' in listdir():
            output_files = listdir('ap_output') 
            file_name = 'experiment_'+exp_name+'_'
            file_name_len = len(file_name)
            exp_ids = [ int(l[file_name_len:-4]) for l in output_files if l[:file_name_len]==file_name]
            if len(exp_ids)>0: exp_id = max(exp_ids)
            else: raise ValueError('No file found') 
        else: raise ValueError('No file found') 
    
    # Load Experiment
    file = open(f'ap_output/{file_name}{exp_id}.obj', 'rb') 
    loaded_experiment = load(file)
    file.close()
    
    return loaded_experiment
