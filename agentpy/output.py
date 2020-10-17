"""

Agentpy
Output Module

Copyright (c) 2020 Joël Foramitti
    
"""

import pandas as pd

from os import listdir, makedirs
import json

from .tools import attr_dict, make_list

class data_dict(attr_dict):
    
    """
    Agentpy dictionary for output data
    
    """
    
    def __init__(self):
        super().__init__() 
        self.log = {}
    
    def __repr__(self):
        
        rep = "data_dict {"
        
        for k,v in self.items():
            
            if isinstance(v,pd.DataFrame): 
                lv = len(list(v.columns))
                rv = len(list(v.index)) 
                rep += f"\n'{k}': DataFrame with {lv} variable{'s' if lv!=1 else ''} and {rv} row{'s' if rv!=1 else ''}"
            elif isinstance(v,dict): 
                lv = len(list(v.keys()))
                rep += f"\n'{k}': Dictionary with {lv} key{'s' if lv!=1 else ''}"
            elif isinstance(v,list): 
                lv = len(v)
                rep += f"\n'{k}': List with {lv} entr{'ies' if lv!=1 else 'y'}"
            else: 
                rep += f"\n'{k}': Object of type {type(v)}"
                
        return rep + "\n}"
    
    def ps_keys(self):
        
        """ Returns the keys of varied parameters """
        
        df = self.parameter_sample
        p_list = []
        
        # creating a list of dataframe columns 
        columns = list(df) 

        for i in columns: 

            # printing the third element of the column 
            if len(df[i].unique()) > 1:
                p_list.append(i)
                
        return p_list

    
    def get_pars(self, parameters = None):
        
        """ Returns pandas dataframe with parameters and run_id """
        
        dfp = pd.concat( [self.parameter_sample] * self.log['iterations'] )
        dfp = dfp.reset_index(drop=True)
        dfp.index.name = 'run_id'
        if parameters is not None and not True: dfp = dfp[parameters] # Select parameters
            
        return dfp  
    
    def get_measures(self,measures=None,parameters=None,reset_index=True):
        
        """ Returns pandas dataframe with measures """
        
        df = self.measures
        
        # Add parameters
        if parameters:
            dfp = self.get_pars(parameters)
            if isinstance(df.index, pd.MultiIndex): dfp = dfp.reindex(df.index,level='run_id')
            df = pd.concat([df,dfp],axis=1)   
            
        return df 
    
    def get_vars(self,var_keys=None,obj_types=None,parameters=None,reset_index=True):

        """ Returns pandas dataframe with variables """
        
        var_keys = make_list(var_keys)
        if parameters is not True:
            parameters = make_list(parameters)
        df_dict = self
        
        # Select variable dataframes
        df_dict = { k[:-5]:v for k,v in df_dict.items() if 'vars' in k} 
        
        # Select object types 
        if var_keys: df_dict = { k:v for k,v in df_dict.items() if any(x in v.columns for x in var_keys)}  
        if obj_types: df_dict = { k:v for k,v in df_dict.items() if k in obj_types} 
            
        # Create dataframe
        df = pd.concat( df_dict ) 
        df.index = df.index.set_names('obj_type',level=0) 
        if var_keys: df = df[var_keys] # Select var_keys

        # Add parameters
        if parameters:
            dfp = self.get_pars(parameters)
            dfp = dfp.reindex(df.index,level='run_id')
            df = pd.concat([df,dfp],axis=1)

        # Reset index
        if reset_index: df = df.reset_index()

        return df
    
    def _last_exp_id(self, name, path):
        
        """ Identifies existing experiment data and return highest id. """
        
        exp_id = 0
        output_dirs = listdir(path)  
        exp_dirs = [s for s in output_dirs if name in s]
        if exp_dirs:
            ids = [int(s.split('_')[-1]) for s in exp_dirs ]
            exp_id = max(ids)
        return exp_id 
           
    def save(self, exp_name = None, exp_id = None, path = 'ap_output', display = True):      

        """ Writes output data to directory ``{path}/{exp_name}_{exp_id}/``.
        
        Arguments:
            exp_name (str, optional): Name of the experiment.
                If none is passed, the name of the experiment instance is used.
            exp_id (int, optional): Number of the experiment.
                If none is passed, the next available id in the target directory is used.
            path (str, optional): Target directory to write files to (default 'ap_output').
            display (bool, optional): Whether to display saving progress (default True).
        
        """
        
        # Create output directory if it doesn't exist
        if path not in listdir(): makedirs(path) 
            
        # Set exp_name
        if exp_name is None: name = self.log['name']
        name = name.replace(" ", "_")
        
        # Set exp_id 
        if exp_id is None:
            exp_id = self._last_exp_id(name, path) + 1
        
        # Create new directory for output
        path = f'{path}/{name}_{exp_id}'
        makedirs(path) 
        
        # Save experiment data
        for key,output in self.items():
            
            t = type(output)
            
            if t == pd.DataFrame: 
                output.to_csv(f'{path}/{key}.csv')
            elif t == dict:
                with open(f'{path}/{key}.json', 'w') as fp:
                    json.dump(output, fp)
            elif t == nx.Graph:
                nx.write_graphml(output, f'{path}/{key}.graphml') 
        
        if display: print(f"Data saved to {path}")

               
    def load(self, exp_name='experiment', exp_id=None, path='ap_output', display = True):
        
        """ Reads output data from directory ``{path}/{exp_name}_{exp_id}/``.
        
        Arguments:
            exp_name (str, optional): Name of the experiment (default 'experiment')
            exp_id (int, optional): Number of the experiment.
                If none is passed, the highest available id in the target directory is used.
            path (str, optional): Target directory to read files from (default 'ap_output').
            display (bool, optional): Whether to display loading progress (default True).
        
        """  
              
        def load_file(self, path, file, display):

            print(f'Loading {file} - ',end='')

            i_cols = ['sample_id','run_id','scenario','env_key','agent_id','t']
            
            ext = file.split(".")[-1]
            key = file[:-(len(ext)+1)]
            
            path = path + file
            
            try: 

                if ext == 'csv': 
                    self[key] = df = pd.read_csv( path )
                    index = [i for i in i_cols if i in df.columns]
                    if index: self[key] = df.set_index(index)
                elif ext == 'json': 
                    with open(path , 'r') as fp: 
                        self[key] = json.load(fp)
                elif ext == 'graphml':
                    self[key] = nx.read_graphml(path)
                else: 
                    if display: print(f"Error: File type '{ext}' not supported")
                    return

                if display: print('Successful')

            except Exception as e: print(f'Error: {e}')        
        
        # Prepare for loading
        exp_name = exp_name.replace(" ", "_")
        if not exp_id: 
            exp_id = self._last_exp_id(exp_name, path)
            if exp_id == 0:
                raise ExperimentError(f"No experiment found with name '{exp_name}' in path '{path}'")         
        path = f'{path}/{exp_name}_{exp_id}/'
        if display: print(f'Loading from directory {path}')    
        
        # Loading data
        for file in listdir(path): load_file(self,path,file,display)
            
        return self
        

def save(data,*args,**kwargs):
    
    """ Alias of ``data.save(*args,**kwargs)`` """
    
    data.save(*args,**kwargs)
    
        
def load(*args,**kwargs):
    
    """ Alias of ``data_dict().load(*args,**kwargs)`` """
    
    return data_dict().load(*args,**kwargs)