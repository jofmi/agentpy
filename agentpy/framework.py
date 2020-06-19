"""

Agentpy 
Framework Module

Copyright (c) 2020 Joël Foramitti

"""

import numpy as np
import pandas as pd
import warnings 



#### Parent Classes ###

class attr_dict(dict):
    
    """ Dictionary where keys and attributes are the same """
    
    def __init__(self):
        super().__init__()
        self.__dict__ = self
        

    
### Custom Functions ###

def make_list(element):
    
    """ Turns element into a list if it is not of type list or tuple """
    
    if not isinstance(element, (list, tuple)): element = [element]
        
    return element
    
    
    
#### Multi-Class Methods ###

def init_vars(self,var_keys,values=0):

    """ Initializes a set of variables """
    
    if not isinstance(var_keys, (list, tuple)): var_keys = [var_keys]
    if not isinstance(values, (list, tuple)): values = [values] * len(var_keys)

    for var_key, value in zip(var_keys,values): self[var_key] = value


def init_recs(self,var_keys):

    """ Declares a set of variables to be recorded """

    var_keys = make_list(var_keys)

    for var_key in var_keys: self.log[var_key] = []


def record(self):
    
    """ Records all variables declared in the log """

    for var_key in self.log.keys():
        
        self.log[var_key].append( self[var_key] )    
    
    

### Main Classes ###

class agent(attr_dict):   
    
    """ 
    
    Agent Class 
    
    Holds actions (methods) and variables (attributes)
    Subclass of attr_dict: Variables can also be accessed as dict entries
    
    """
    
    
    # Import Multi-Class Methods
    init_vars, init_recs, record = init_vars, init_recs, record
    
    
    def __init__(self,model,env_keys=[]):
        
        super().__init__() 
        
        self.model = model
        self.p = model.p      
        self.type = type(self).__name__
        self.t0 = model.t # Time of birth
        self.log = {}     
        
        self.id = model.get_id()
        self.env = model.envs[env_keys[0]] # Main environment   
        self.env_keys = env_keys # All environments
        
        if 'init' in dir(self): self.init() # Run init() if defined in subclass
            
        
    def remove(self, env_keys=None):
        
        """ Removes the agent from 'env_keys' (default None) """
            
        if self.env_keys == []: 
            warnings.warn("Failed to remove agent that is already removed from all environments")
            return

        if not env_keys: env_keys = self.env_keys # Select all environments by default
        
        else: env_keys = make_list(env_keys)
        
        for env_key in env_keys: 
            self.env_keys.remove(env_key)
            env = self.model.envs[env_key] # Remove from master environment
            env.agents.remove(self) 
            env.agents_by_type[self.type].remove(self)
        
        # Update main environment
        if self.env.key in env_keys: 
            if self.env_keys: self.env = model.envs[self.env_keys[0]] 
            else: self.env = None
    
    
class environment(attr_dict):
    
    """ 
    
    Environment Class 
    
    Holds a list of agents, as well as environment-specific actions (methods) and variables (attributes)
    Subclass of attr_dict: Attributes can also be accessed as dict entries
    
    """

    
    # Import Multi-Class Methods
    init_vars, init_recs, record = init_vars, init_recs, record
    
    
    def __init__(self,model,env_key):
        
        super().__init__() 
        
        self.model = model
        self.p = model.p    
        self.type = type(self).__name__
        self.t0 = model.t # Time of birth
        self.log = {}
        
        self.key = env_key
        self.agents = []
        self.agents_by_type = {}
    
        if 'init' in dir(self): self.init()
          
    
    def add_agents( self, n = 0, agent_class = agent, **kwargs ):
        
        """ Adds 'n' new agents of class 'agent_class' to this environment """ 
        
        # Create new agent category if agent_class is new
        agent_type = agent_class.__name__
        if agent_type not in self.agents_by_type: 
            self.agents_by_type[ agent_type ] = []
            
        for i in range(n): 

            new_agent = agent_class( self.model , [self.key] , **kwargs )

            self.agents.append( new_agent )
            self.agents_by_type[ agent_type ].append( new_agent )
            self.model.agents.append( new_agent ) # Master environment
        
        
    def action(self, action_key, agent_type = None, random=False ): 
        
        """ 
        
        Calls an action for selected agents in this environment
        
        Arguments:
            action_key (string): name of the agent action to be called 
            agent_type (string): specifies the type of agents (default None)
            random (bool): shuffles the order of agents if True (default False)

        """
           
        if agent_type: agents = self.agents_by_type[agent_type]
        else: agents = self.agents
        
        if random == True: 
            
            agents = list( agents ) # Shallow Copy
            np.random.shuffle( agents )
            
        for agent in agents: getattr( agent, action_key )()   

            
    def count(self, agent_type):
        
        """ Returns number of agents of type 'agent_type' in this environment """
        
        return len( self.agents_by_type[agent_type] )
    
    
    def select_random( self, n=1 , agent_type=None ):
    
        """ Select 'n' (default 1) random agents of type 'agent_type' (default None) """
        
        if agent_type: agents = self.agents_by_type[agent_type]
        else: agents = self.agents
        
        if agents == []: return []  
        elif n == 1: return np.random.choice(agents,n)[0] # Return element, not list
        else: return np.random.choice(agents,n)
    
    
    def select_id( self, agent_ids ):
                
        """ Select agents with id in agent_ids """
        
        return [ a for a in self.agents if a.id in make_list(agent_ids) ]                    
       
    
    
class model():
    
    """ 
    
    Model Class 
    
    Holds model parameters, a dictionary of the model environments, and a list of all agents
    The method run() performs the simulation, which generates the dataframes 'output' and 'measures'
    
    """
 

    def __init__(self,parameters,run_id=0,scenario='default'):
        
        self.agents = []
        self.envs = attr_dict() 
        self.p = attr_dict() 
        self.p.update(parameters)
        
        self.run_id = run_id
        self.scenario = scenario
        
        self.t = 0 
        self.agent_id_counter = 0  
  
        self.log = {}
        self.output = None
        self.measures = None
        
        
    def get_id( self ):
    
        """ Returns unique id for a new agent """
        
        agent_id = self.agent_id_counter
        self.agent_id_counter += 1 
        return agent_id
    
    
    def add_environment( self, env_key, env_class=environment, return_env = True, **kwargs ):
        
        """ Creates and returns a new environment of class 'env_class' with the name 'env_key' """
        
        self.envs[env_key] = env_class(self,env_key,**kwargs)
        return self.envs[env_key]
    
    
    def add_agents( self, n, agent_class=agent, env_keys=[], **kwargs ):
        
        """ Adds 'n' new agents of class 'agent_class' to environments specified in 'env_keys' """ 
        
        env_keys = make_list(env_keys)
        
        for i in range(n): 
            
            new_agent = agent_class( self, list(env_keys), **kwargs) # Shallow copy of env_keys
            self.agents.append( new_agent ) # Add agent to master environment
            
            for env_key in env_keys:
                
                env = self.envs[env_key]
                
                # Create new agent category if agent_class is new
                if agent_class.__name__ not in env.agents_by_type: 
                    env.agents_by_type[ agent_class.__name__ ] = []

                env.agents.append( new_agent )
                env.agents_by_type[ agent_type ].append( new_agent )
    
    
    def action(self, action_key, agent_type = None, env_keys = None, random = False):
        
        """ Calls 'action' for agents in selected environments """
        
        if not env_keys: env_keys = self.envs
        env_keys = make_list(env_keys)
        
        for env_key in env_keys:
            self.envs[env_key].action(action_key,agent_type,random) 
    
    
    def step(self):
        
        """ This method has to be over-written to define the model's action per simulation step """
        
        warnings.warn(f"Method step() is undefined")
        
        
    def evaluation(self):
        
        """ This method can be over-written to record measures at the end of the simulation """
        
        pass
        
        
    def run(self):
        
        """ Performs the agent-based simulation """
        
        if 'init' in dir(self): self.init() # Initialize model
        
        self.record() # Record round zero
        
        for i in range(self.p.steps):
            
            self.t += 1
            self.step()
            self.record()
        
        self.evaluation()
        
        self.create_output()
           
        return self.output
    
    
    def record(self):
        
        """ Records variables declared in the logs of all agents and environments """
        
        for env in self.envs.values(): env.record()
        for agent in self.agents: agent.record()
            
            
    def rec_measure(self, measure_key, value):
        
        """ Records an evaluation measure with name 'measure_key' and 'value' """
        
        self.log[measure_key] = value
 

    def create_output(self):
        
        """ Generates the dataframes 'output' and 'measures' """
        
        # PART 1 - Dynamic Output
        
        output_list = []
        
        for env in self.envs.values():
            
            for var_key, values in env.log.items():
                
                for t,value in enumerate(values, start = env.t0 ):
                    
                    output_list.append( [self.run_id, self.scenario, var_key, env.key, np.NaN , t , value] )
                    
            for agent in env.agents:
                
                for var_key, values in agent.log.items():
                
                    for t,value in enumerate(values, start = agent.t0 ):
                    
                        output_list.append( [self.run_id, self.scenario, var_key, env.key, agent.id , t , value] ) 
        
        # Convert output_list to dataframe. Format: run_id, scenario, var_key, env_key, agent_id, time, value
        self.output = pd.DataFrame(output_list, columns = ['run_id','scenario','var_key','env_key','agent_id','t','value'] ) 

        # PART 2 - Evaluation Measures
        
        measure_list = []        
            
        for measure_key, value in self.log.items():

            measure_list.append( [self.run_id, self.scenario, measure_key, value] )
        
        # Convert measure_list to dataframe. Format: run_id, scenario, measure_key, value
        self.measures = pd.DataFrame(measure_list, columns = ['run_id','scenario','measure_key','value'] )
        
        
