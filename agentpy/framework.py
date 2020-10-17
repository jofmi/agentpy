"""

Agentpy 
Framework Module

Copyright (c) 2020 Joël Foramitti

"""

import pandas as pd
import networkx as nx

import random
import operator
import warnings 

from datetime import datetime
from .output import data_dict
from .tools import make_list, attr_dict, attr_list, AgentpyError

### Tools for all classes ###

def record(self, var_keys, value = None):
    
    """ Records an objects variables. 
    
    Arguments:
        var_keys(str or list): Names of the variables
        value(optional): Value to be recorded.
            If no value is given, keys are used to look up the objects attribute values.
    """
    
    for var_key in make_list(var_keys):
    
        # Create empty list if var_key is new (!) should have none until t
        if var_key not in self.log: self.log[var_key] = [None] * len(self.log['t'])

        if self.model.t not in self.log['t']:

            # Create empty slot for new documented time step
            for v in self.log.values(): v.append(None)

            # Store time step
            self.log['t'][-1] = self.model.t

        if value is None: v = self[var_key]
        else: v = value 

        self.log[var_key][-1] = v     
    
def setup(self):
    """Defines the object's actions after creation.
    Can be overwritten and used for initialization."""
    pass  


    
### Level 3 - Agent class ###    

class agent(attr_dict):   
    
    """ 
    
    Individual agent of an agent-based model.
    This class can be used as a parent class for custom agent types.  
    Attributes can be accessed as items (see :class:`attr_dict`).
    Methods of :class:`agent_list` can also be used for individual agents.
    
    Attributes:
        model (model): The agents' model 
        envs (env_dict): The agents' environments
        p (attr_dict): The models' parameters
        log (dict): The agents' recorded variables
        type (str): The agents' type 
        t0 (int): Time-step of the agent's creation
        id (int): The agent's unique identifier
        
    Arguments:
        model (model): The agents' model 
        envs (dict or env_dict, optional): The agents' environments
    
    """
    
    setup = setup
    record = record
    
    def __init__(self,model,envs=None):
        
        super().__init__() 
        
        self.model = model
        self.p = model.p 
        self.t0 = model.t 
        self.id = model._get_id() 
        
        self.envs = env_dict(model)
        if envs: self.envs.update(envs)
        
        self.log = {'t':[]} 
        self.type = type(self).__name__
        
        self.setup() # Custom initialization
        
    def __repr__(self): 
        s = f"Agent {self.id}"
        if self.type != 'agent': s += f" ({self.type})"
        return s
    
    def __hash__(self):
        return self.id # Necessary for networkx
    
    def neighbors(self,key=None):
        
        """ Returns the agents' neighbor's in a network
        
        Arguments:
            key(str, optional): Name of the network. 
                At default, the first network in the agents' environments is selected.
        """
        
        # Select network
        if key == None: 
            for k,v in self.envs.items():
                if v.topology == 'network':
                    key = k
                    break
        elif key not in self.envs.keys(): AgentpyError(f"Agent {self.id} has no network '{key}'")
        if key == None: AgentpyError(f"No network found for agent {self.id}")
        
        return agent_list([n for n in self.envs[key].neighbors(self)]) # (!) list comprehension necessary?
    
    def remove(self,keys=None):
        
        """ Removes the agent from environments or the model 
        
        Arguments:
            keys(str or list, optional): Environments from which the agent should be removed.
                If no keys are given, agent is removed from the model completely.
        """
        
        if keys is None: 
            envs = self.envs # Select all environments by default
            self.model.agents.remove(self) # Delete from model    
        else: 
            envs = { k:v for k,v in self.envs if k in keys}
            
        for key,env in envs.items(): 
            env.agents.remove(self)
            del self.envs[key]

            
class agent_list(attr_list):
    
    """ List of agents. 
    Attribute access (e.g. `agent_list.x`) is forwarded to each agent and returns a list of attribute values.
    This also works for method calls (e.g. `agent_list.x()`) , which returns a list of method return values.
    Assignments of attributes (e.g. `agent_list.x = 1`)  are forwarded to each agent in the list.
    Basic operators can also be used (e.g. `agent_list.x += 1` or `agent_list.x = agent_list.y * 2`).
    Comparison operators (e.g. `agent_list.x == 1`)  return a new agent_list with agents that fulfill the condition.
    See :func:`attr_list` for more information.
    
    Arguments:
        agents(agent or list, optional): Initial list entries
    """
    
    def __init__(self, agents = None):
        super().__init__()
        if agents: self.extend( make_list(agents) )
    
    def __repr__(self): 
        return f"<list of {len(self)} agents>"

    def do(self, method, *args, order=None, return_value=False, **kwargs): 
        
        """ 
        Calls a method for all agents in the list
        
        Arguments:
            method (str): Name of the method
            order (str, optional): If 'random', order in which agents are called will be shuffled
            *args: Will be forwarded to the agents method
            **kwargs: Will be forwarded to the agents method
        """
        
        agents = self
        
        if order == 'random': 
            agents = list( self ) # Copy
            random.shuffle( agents )
                
        for agent in agents: 
            getattr( agent, method )(*args,**kwargs)  
    
    def select(self,var_key,value=True,relation="=="):
        
        """ Returns a new `agent_list` of selected agents.
        
        Arguments:
            var_key (str): Variable for selection
            value (optional): Value for selection, default `True`
            relation (str, optional): Relation between variable and value
                Options are '=='(default),'!=','<','<=','>','>=' """
        
        relations = {'==':operator.eq,'!=':operator.ne,
                     '<':operator.lt,'<=':operator.le,
                     '>':operator.gt,'>=':operator.ge}
        
        return agent_list([a for a in self if relations[relation](a[var_key],value)])
    
    def assign(self,var_key,value):
        
        """ Assigns value to var_key for each agent."""
        
        for agent in self: setattr(agent,var_key,value)
                        
    def of_type(self,agent_type):
        
        """ Returns a new `agent_list` with agents of agent_type """
                        
        return self.select('type',agent_type)
    
    def random(self,n=1):
        
        """ Returns a new `agent_list` with n random agents """
        
        return agent_list(random.sample(self,n))
           
    
### Level 2 - Environment class ###        
    
def add_agents(self, agents, agent_class=agent, **kwargs):
        
    """ 
    Adds agents to the environment.
    
    Arguments:
        agents(int or agent_list): Number of new agents or list of existing agents.
        agent_class(class): Type of agent to be created if int is passed for agents.
    """ 
    
    # (!) what if single agent is passed? 

    if type(agents) == agent_list: # (!) or normal iterable with agents?

        new_agents = agents
        # (!) need to add new environment to agent
        
    elif type(agents) == int:
        
        new_agents = agent_list()
        
        for i in range(agents): 
            new_agents.append(agent_class(self.model, {self.key:self}, **kwargs))
                 
    else: raise ValueError("agents must be agent_list or int") # (!) improve error
        
    # Add agents to this env
    self.agents.extend( new_agents )

    # Add agent to master list
    if self != self.model: 
            self.model.agents.extend( new_agents )

    return new_agents
    
def _env_init(self,model,key):
    
    attr_dict.__init__(self)
    self.model = model
    self.agents = agent_list()
    self.p = model.p          
    self.key = key
    self.type = type(self).__name__
    self.t0 = model.t 
    self.log = {'t':[]}

class environment(attr_dict):
    
    """ Standard environment that contains agents without a topology.
    This class can be used as a parent class for custom environment types. 
    Attributes can be accessed as items (see :class:`attr_dict`).
    
    Attributes:
        model (model): The environments' model 
        agents (agent_list): The environments' agents
        p (attr_dict): The models' parameters
        type (str): The environments' type 
        key(str): The environments' name
        topology(str): Topology of the environment
        t0 (int): Time-step of the environments' creation
        log (dict): The environments' recorded variables
        
    Arguments:
        model (model): The environments' model 
        key (dict or env_dict, optional):  The environments' name
    
    """
    
    add_agents = add_agents
    record = record
    setup = setup
    
    def __init__(self,model,key):
        
        _env_init(self,model,key)
        self.topology = None
        self.setup()
        
    def __repr__(self): 
        return f"<environment '{self.key}' of type '{self.type}'>"
        
        
class network(environment): 
    
    """ Network environment that contains agents with a graph topology.
    Unknown attribute and method calls are forwarded to the graph.
    Inherits attributes and methods from :class:`environment`.
    
    Attributes:
        graph(networkx.Graph): Networkx Graph instance
    
    Arguments:
        model (model): The environments' model 
        key (dict or env_dict, optional):  The environments' name
        graph(networkx.Graph): The environments' graph
    """
  
    def __init__(self,model,env_key,graph=None):
          
        _env_init(self,model,env_key)
        self.topology = 'network'
        if graph: self.graph = graph
        else: self.graph = nx.Graph()
            
        self.setup()
        
    def add_agents(self, agents, agent_class=agent, map_to_nodes=False, **kwargs):
        
        """ 
        Adds agents to the network environment.

        Arguments:
            agents(int or agent_list): Number of new agents or list of existing agents.
            agent_class(class): Type of agent to be created if int is passed for agents.
            map_to_nodes(bool,optional): Map new agents to each node of the graph (default False).
                Should be used if a graph with empty nodes has been passed at network creation.
        """         

        new_agents = add_agents(self,agents,agent_class,**kwargs)
        
        if map_to_nodes:
            
            # Map each agent to a node of the graph
            
            if len(new_agents) != len(self.graph.nodes):
                raise ValueError(f"Number of agents ({len(new_agents)}) does not match number of nodes ({len(self.graph.nodes)})")
                
            mapping = { i : agent for i,agent in enumerate(new_agents) }
            nx.relabel_nodes( self.graph , mapping = mapping, copy=False )
            
        else:
            # Add agents to graph
            for agent in new_agents:
                self.graph.add_node( agent ) 
        
    def __getattr__(self, method_name):
        
        # Forward unknown method call to self.graph
        
        def method(*args, **kwargs):
            return getattr(self.graph, method_name)(*args, **kwargs)
        
        try: return method
        except AttributeError:
            raise AttributeError(f"module {__name__} has no attribute {name}")
        
class env_dict(dict):  
    
    """ Dictionary for environments
    
    Attributes:
        model(model): The current model instance
    """

    def __init__(self,model):
        super().__init__()
        self.model = model    
            
    def of_type(self,env_type):
        
        """ Returns an env_dict with environments of `env_type`"""
        
        new_dict = env_dict(self.model)
        selection = {k : v for k, v in self.items() if v.type == env_type}
        new_dict.update(selection)
        return new_dict
    
    def do(self,action):
        
        """ Calls `action` for all environments """
        
        for env in self.values(): getattr( env, action )() 
    
    def add_agents(self,*args,**kwargs):
        
        """ Adds agents to all environments in self """
        
        # Depreciated (!)
        
        new_agents = agent_list(self.model)
        new_agents.add(*args,**kwargs)
        
        for env in self.values():
            env.agents.add(new_agents)


### Level 1 - Model Class ###
    
class model(attr_dict):
    
    """ An agent-based model.
    This class can be used as a parent class for custom models. 
    Attributes can be accessed as items (see :class:`attr_dict`).
    Environments can be called as attributes by their keys.
    
    Attributes:
        model(model): Reference to self
        envs(env_dict): The models' environments
        agents(agent_list): The models' agents
        p(attr_dict): The models' parameters
        type(str): The models' name
        t (int): Current time-step of the model
        log (dict): The models' recorded variables
        output(data_dict): Simulation output data
        
    Arguments:
        parameters(dict,optional): Model parameters
        run_id(int,optional): Number of current run
        scenario(str,optional): Current scenario
    """
    
    setup = setup
    record = record
    add_agents = add_agents
    
    def __init__(self,
                 parameters = None,
                 run_id = None,
                 scenario = None
                ):
        
        super().__init__() 
        
        self.model = self
        self.key = 'model'
        self.envs = env_dict(self) 
        self.agents = agent_list()
        self.p = attr_dict() 
        if parameters: self.p.update(parameters)
        
        self.type = type(self).__name__
        self.run_id = run_id
        self.scenario = scenario
        
        # Counters
        self.t = 0 
        self._id_counter = 0  
        
        # Recording
        self.log = {'t':[]}
        self.measure_log = {}
        self.output = data_dict()
        self.output.log['name'] = self.type
        self.output.log['time_stamp'] = str(datetime.now())
        
        # List of non-variable keys
        self._int_keys = list(self.keys())
 
    def _get_id(self):
        
        self._id_counter += 1 
        return self._id_counter - 1   
    
    def __getattr__(self, name): 
            
        try: # Access environments
            return self.envs[name]  
        except KeyError:
            raise AttributeError(f"Model '{self.type}' has no attribute or environment '{name}'") 
    
    # Creation & Destruction

    def add_env(self, env_key, env_class=environment, **kwargs):

        """ Creates and returns a new environment """

        self.envs[env_key] = env_class(self.model, env_key, **kwargs) 
        return self.envs[env_key] 
    
    def add_envs(self,env_keys,*args,**kwargs):
        for env_key in env_keys: self.add_env(env_key,*args,**kwargs)
                    
    def add_network(self, env_key, graph = None, env_class=network , **kwargs):
        self.add_env(env_key, env_class=network, graph=graph, **kwargs)
            
    def add_networks(self,env_keys,*args,**kwargs):      
        for env_key in env_keys: self.add_network(env_key,*args,**kwargs)
            
    
    # Recording functions
            
    def measure(self, measure_key, value):
        
        """ Records an evaluation measure """
        
        self.measure_log[measure_key] = [value]
        
    def record_graph(self, graph_keys):  
        
        """ Records a network """ # (!) unfinished
        
        for graph_key in make_list(graph_keys):
            
            G = self.envs[graph_key].graph
            H = nx.relabel_nodes(G, lambda x: x.id)
            # (!) graph attributes? H.graph.t = self.t
            if 'graphs' not in self.model.output.keys():
                self.model.output['graphs'] = [] #{'t':[]}
            self.model.output['graphs'].append(H)
            #if graph_key not in self.model.output['graphs'].keys():
            #    self.model.output['graphs'][graph_key] = []
            #self.model.output['graphs'][graph_key].append(H)
            #self.model.output['graphs']['t'].append(self.t)
    
    def record_all(self):
        
        """ Records all model variables """
        
        keys = list(self.keys())
        for key in self._int_keys:
            keys.remove(key)
        
        self.record(keys)
    
    # Main simulation functions 
    
    def step(self):
        """ 
        Defines the model's actions during each simulation step.
        Can be overwritten and used to perform the models' main dynamics.
        """
        pass
    
    def update(self):
        """ 
        Defines the model's actions after setup and each simulation step.
        Can be overwritten and used for the recording of dynamic variables.
        """
        pass  
    
    def end(self):
        """ 
        Defines the model's actions after the last simulation step.
        Can be overwritten and used for final calculations and the recording of evaluation measures.
        """
        pass
    
    def stop_if(self):
        """ 
        Stops the simulation if return value is `False`. 
        Returns whether time-step 't' has reached the parameter 'steps'.
        Can be overwritten with a custom function.
        """
        return self.t >= self.p.steps 
    
    def stop(self):
        """ Stops the simulation. """
        self.stopped = True       
            
    def run(self,display=True):
        
        """ Executes the simulation of the model.
        
        The order of events is as follows:
        
        - setup()
        - update()
        - while stop_if() returns False and stop() hasn't been called:
            - t += 1
            - step()
            - update()
        - end()
        
        Arguments:
            display(bool,optional): Whether to display simulation progress (default True).
        """
        
        t0 = datetime.now() # Time-Stamp
        
        self.stopped = False
        self.setup() 
        self.update()
        
        while not self.stop_if() and not self.stopped:
            
            self.t += 1
            self.step()
            self.update()
            
            if display: print( f"\rCompleted: {self.t} steps" , end='' ) 
        
        self.stopped = True
        self.end() 
        self.create_output()
        self.output.log['run_time'] = ct = str( datetime.now() - t0 )
        self.output.log['steps'] = self.t         
        
        if display: print(f"\nRun time: {ct}\nSimulation finished")
        
        return self.output
    
    def create_output(self):
        
        """ Generates an 'output' dictionary out of object logs """
    
        def create_df(obj,c2):
            
            df = pd.DataFrame(obj.log)
            df['obj_id'] = obj[c2] # obj_id
            return df
            
        def output_from_obj_list(self,obj_list,c1,c2,columns):
            
            """ Create output for agent_list or env_dict """
            keys = []
            obj_types = {}
            
            for obj in obj_list:
                if obj.log != {'t':[]}:
                    
                    # Category (obj_type)
                    obj_type = f'{obj.type}_vars'
                    obj.log['obj_id'] = [obj[c2]] * len(obj.log['t'])
                    
                    if obj_type not in obj_types.keys():
                        obj_types[obj_type] = {}
                    
                    for k,v in obj.log.items():
                        if k not in obj_types[obj_type]:
                            obj_types[obj_type][k] = []
                        obj_types[obj_type][k].extend(v)
                    
            # Once per type
            for obj_type, log in obj_types.items():
                df = pd.DataFrame( log )
                for k,v in columns.items(): df[k]=v
                df = df.set_index(list(columns.keys())+['obj_id','t'])
                self.output[obj_type] =  df 
        
        # Additional columns 
        columns = {}
        if self.run_id is not None: columns['run_id'] = self.run_id
        if self.scenario is not None: columns['scenario'] = self.scenario
        
        # Create object var output
        output_from_obj_list(self, self.agents, 'agent', 'id', columns)
        output_from_obj_list(self, self.envs.values(), 'env', 'key', columns)
        
        # Create model var output
        if self.log != {'t':[]}: 
            df = pd.DataFrame(self.log)
            df['obj_id'] = 'model'
            for k,v in columns.items(): df[k]=v
            df = df.set_index(list(columns.keys())+['obj_id','t'])
            self.output['model_vars'] = df
        
        # Create measure output
        if self.measure_log != {}:   
            d = self.measure_log
            for key,value in columns.items(): d[key] = value
            df = pd.DataFrame(d)
            if columns: df = df.set_index(list(columns.keys()))
            self.output['measures'] = df