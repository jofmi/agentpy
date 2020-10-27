"""

Agentpy 
Framework Module

Copyright (c) 2020 Joël Foramitti

"""

import pandas as pd
import networkx as nx

import random as rd
import operator
import warnings 

from datetime import datetime
from itertools import product
from .output import data_dict
from .tools import make_list, attr_dict, obj_list, nested_list, AgentpyError

### Tools for all classes ###

def record(self, var_keys, value = None):
    
    """ Records an objects variables. 
    
    Arguments:
        var_keys(str or list): Names of the variables
        value(optional): Value to be recorded.
            If no value is given, var_keys have to refer to existing object attributes.
    """
    
    for var_key in make_list(var_keys):
    
        # Create empty lists
        if 't' not in self.log: 
            self.log['t'] = []
        if var_key not in self.log: 
            self.log[var_key] = [None] * len(self.log['t'])

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
    
    """ Individual agent of an agent-based model.
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
        
        self.log = {} 
        self.type = type(self).__name__
        
        self.setup() # Custom initialization
        
    def __repr__(self): 
        s = f"Agent {self.id}"
        if self.type != 'agent': s += f" ({self.type})"
        return s
    
    def __hash__(self):
        return self.id # Necessary for networkx
    
    def pos(self,key=None):
        
        # Select network
        if key == None: 
            for k,v in self.envs.items():
                if v.topology == 'grid':
                    key = k
                    break      
        
        return self.envs[key].pos[self]
    
    def neighbors(self,key=None):
        
        """ Returns the agents' neighbor's in a network
        
        Arguments:
            key(str, optional): Name of the network. 
                At default, the first network in the agents' environments is selected.
        """
        
        # Select network
        if key == None: 
            for k,v in self.envs.items():
                if v.topology == 'network' or v.topology == 'grid':
                    key = k
                    break                    
        elif key not in self.envs.keys(): AgentpyError(f"Agent {self.id} has no network '{key}'")
        if key == None: AgentpyError(f"No network found for agent {self.id}") # (!) Faulty logic
        
        return agent_list(self.envs[key].neighbors(self))
    
    def leave(self,keys=None):
        
        """ Removes the agent from environments or the model 
        
        Arguments:
            keys(str or list, optional): Environments from which the agent should be removed.
                If no keys are given, agent is removed from the model completely.
        """
        
        if keys is None: 
            envs = self.envs # Select all environments by default
            self.model.agents.remove(self) # Delete from model    
        else: 
            envs = { k:v for k,v in self.envs.items() if k in keys}
            
        for key,env in envs.items(): 
            env.agents.remove(self)
            del self.envs[key]

            
class agent_list(obj_list):
    
    """ List of agents. 
    
    Attribute access (e.g. `agent_list.x`) is forwarded to each agent and returns a list of attribute values.
    This also works for method calls (e.g. `agent_list.x()`) , which returns a list of method return values.
    Assignments of attributes (e.g. `agent_list.x = 1`)  are forwarded to each agent in the list.
    Basic operators can also be used (e.g. `agent_list.x += 1` or `agent_list.x = agent_list.y * 2`).
    Comparison operators (e.g. `agent_list.x == 1`)  return a new agent_list with agents that fulfill the condition.
    See :func:`obj_list` for more information.
    
    Arguments:
        agents(agent or list of agent, optional): Initial agents of the list
    """
    
    def __init__(self, agents = None):
        super().__init__()
        if agents: self.extend( make_list(agents) )
    
    def __repr__(self): 
        return f"<agent_list with {len(self)} agents>"

    def do(self, method, *args, shuffle = False, **kwargs): 
        
        """ 
        Calls ``method(*args,**kwargs)`` for every agent.
        
        Arguments:
            method (str): Name of the method
            *args: Will be forwarded to the agents method
            random (bool, optional): Whether to shuffle order in which agents are called (default False)
            **kwargs: Will be forwarded to the agents method
        """
        
        agents = self
        
        if order == 'random': 
            agents = list( self ) # Copy
            rd.shuffle( agents )
                
        for agent in agents: 
            getattr( agent, method )(*args,**kwargs)  
    
    def select(self,var_key,value=True,relation="=="):
        
        """ Returns a new :class:`agent_list` of selected agents.
        
        Arguments:
            var_key (str): Variable for selection
            value (optional): Value for selection (default True)
            relation (str, optional): Relation between variable and value (default '==').
                Options are '==','!=','<','<=','>','>=' """
        
        relations = {'==':operator.eq,'!=':operator.ne,
                     '<':operator.lt,'<=':operator.le,
                     '>':operator.gt,'>=':operator.ge}
        
        return agent_list([a for a in self if relations[relation](a[var_key],value)])
    
    def assign(self,var_key,value):
        
        """ Assigns ``value`` to ``var_key`` for each agent."""
        
        for agent in self: setattr(agent,var_key,value)
                        
    def of_type(self,agent_type):
        
        """ Returns a new :class:`agent_list` with agents of type ``agent_type``. """
                        
        return self.select('type',agent_type)
    
    def random(self,n=1):
        
        """ Returns a new :class:`agent_list` with ``n`` random agents (default 1)."""
        
        return agent_list(rd.sample(self,n))
           
    
### Level 2 - Environment class ###        
    
def add_agents(self, agents, agent_class=agent, **kwargs):
        
    """ 
    Adds agents to the environment.
    
    Arguments:
        agents(int or agent_list): Number of new agents or list of existing agents.
        agent_class(class): Type of agent to be created if int is passed for agents.
    """ 
        
    if isinstance(agents,int):
        
        # Create new agent instances
        new_agents = agent_list([agent_class(self.model, {self.key:self}, **kwargs) for _ in range(agents)])   
        
        # Add agents to master list
        if self != self.model:    
            self.model.agents.extend( new_agents )
    
    else:
        
        # Add existing agents
        if isinstance(agents,agent_list): new_agents = agents
        else: new_agents = agent_list(agents)
        
        # Add new environment to agents
        if self != self.model: 
            for agent in new_agents:
                agent.envs[self.key] = self
        
    # Add agents to this env
    self.agents.extend( new_agents )

    return new_agents
    
def _env_init(self,model,key):
    
    attr_dict.__init__(self)
    self.model = model
    self.agents = agent_list()
    self.p = model.p          
    self.key = key
    self.type = type(self).__name__
    self.t0 = model.t 
    self.log = {}

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
            agents(int or agent or list or agent_list): Number of new agents or existing agents.
            agent_class(class): Type of agent to be created if int is passed for agents.
            map_to_nodes(bool,optional): Map new agents to each node of the graph (default False).
                Should be used if a graph with empty nodes has been passed at network creation.
        """         
        
        # Standard adding
        new_agents = add_agents(self,agents,agent_class,**kwargs)
        
        # Extra network features
        if map_to_nodes: 
            # Map each agent to a node of the graph
            if len(new_agents) != len(self.graph.nodes):
                raise ValueError(f"Number of agents ({len(new_agents)}) does not match number of nodes ({len(self.graph.nodes)})")
            mapping = { i : agent for i,agent in enumerate(new_agents) }
            nx.relabel_nodes( self.graph , mapping = mapping, copy=False )
        else:
            # Add agents to graph as new nodes
            for agent in new_agents:
                self.graph.add_node( agent ) 
    
    def neighbors(self,agent):
        
        """ Returns :class:`agent_list` of agents that are connected to the passed agent. """
        
        return agent_list([n for n in self.graph.neighbors(agent)])
    
    def __getattr__(self, method_name):
        
        # Forward unknown method call to self.graph
        
        def method(*args, **kwargs):
            return getattr(self.graph, method_name)(*args, **kwargs)
        
        try: return method
        except AttributeError:
            raise AttributeError(f"module {__name__} has no attribute {name}")

class grid(environment): 
    
    """ Grid environment that contains agents with a spatial topology.
    Inherits attributes and methods from :class:`environment`.
    
    Attributes:
        pos(dict): Agent positions
    
    Arguments:
        model (model): The environments' model 
        key (dict or env_dict, optional):  The environments' name
        dim(int, optional): Number of dimensions (default 2).
        size(int or tuple): Size of the grid.
            If int, the same length is assigned to each dimension.
            If tuple, one int item is required per dimension.
    """
  
    def __init__(self,model,env_key,shape):
          
        _env_init(self,model,env_key)
        
        self.topology = 'grid'
        self.grid = nested_list( make_list(shape) , lambda:agent_list() )
        self.shape = shape
        self.pos = {}
            
        self.setup()
        
    def neighbors(self,agent,shape='diamond'):
        
        """ Return agent neighbors """
        
        if shape == 'diamond': return self._get_neighbors4(self.pos[agent],self.grid) 
        elif shape == 'square': return self._get_neighbors8(self.pos[agent],self.grid) 
    
    def area(self,area):
        
        return agent_list( self._get_area(area,self.grid) )
    
    def _get_area(self,area,grid):
        
        """ Return agents in area of style: [(x_min,x_max),(y_min,y_max),...]"""
        
        subgrid = grid[area[0][0]:area[0][1]+1]
        
        if isinstance(subgrid[0],agent_list): # Detect last row (must have agent_lists)
            return [y for x in subgrid for y in x] # Flatten list of agent_lists to list of agents
        
        objects = []
        for row in subgrid: 
            objects.extend( self._get_area(area[1:],row) ) 
            
        return objects
    
    def _get_neighbors4(self,pos,grid,dist=1):
        
        """ Return agents in diamond-shaped area around pos """
        
        subgrid = grid[max(0,pos[0]-dist):pos[0]+dist+1]
        
        if len(pos) == 1: 
            return [y for x in subgrid for y in x] # flatten list
        
        objects = []
        for row,dist in zip(subgrid,[0,1,0]): 
            objects.extend( self._get_neighbors4(pos[1:],row,dist) )
            
        return objects    
    
    def _get_neighbors8(self,pos,grid):
        
        subgrid = grid[max(0,pos[0]-1):pos[0]+2]
        
        if len(pos) == 1: 
            return [y for x in subgrid for y in x] # flatten list
        
        objects = []
        for row in subgrid: 
            objects.extend( self._get_neighbors8(pos[1:],row) )
            
        return objects
    
    def _get_pos(self,grid,pos):
        if len(pos) == 1: return grid[pos[0]]
        return self._get_pos(grid[pos[0]],pos[1:]) 
    
    def get_pos(self,pos):
        return self._get_pos(self.grid,pos)
    
    def change_pos(self,agent,new_position):
        self.get_pos(self.position[agent],self.grid).drop(agent) # Remove from old position
        self.get_pos(new_position,self.grid).append(agent) # Add to new position
        self.pos[agent] = position # Log Position
    
    def add_agents(self, agents, agent_class=agent, positions=None, random = False, map_to_grid=False, **kwargs):
        
        """ Adds agents to the grid environment."""         
        
        # (!) unfinished
        
        # Standard adding
        new_agents = add_agents(self,agents,agent_class,**kwargs)
        
        # Extra grid features
        if map_to_grid:
            pass #(!)
        elif positions:
            for agent,pos in zip(new_agents,positions):
                self.pos[agent] = pos
        elif random:
            positions = list(product(*[ range(i) for i in self.shape ]))
            sample = rd.sample(positions,agents)
            
            for agent,pos in zip(new_agents,sample):
                self.pos[agent] = pos  # Log Position
                self.get_pos(pos).append(agent) # Add to new position

        else:
            for agent in new_agents:
                self.pos[agent] = [0] * self.dim
            
            
class env_dict(dict):  
    
    """ Dictionary for environments
    
    Attributes:
        model(model): The current model instance
    """

    def __init__(self,model):
        
        super().__init__()
        self.model = model    
    
    def __repr__(self):
        
        return f"<env_dict with {len(self.keys())} environments>"
    
    def of_type(self,env_type):
        
        """ Returns :class:`env_dict` of selected environments. """
        
        new_dict = env_dict(self.model)
        selection = {k : v for k, v in self.items() if v.type == env_type}
        new_dict.update(selection)
        return new_dict
    
    def do(self,method,*args,**kwargs):
        
        """ Calls ``method(*args,**kwargs)`` for all environments. """
        
        for env in self.values(): getattr(env, method)(*args, **kwargs) 
    
    def add_agents(self,*args,**kwargs):
        
        """ Calls :meth:`environment.add_agents` with `*args,**kwargs` and forwards new agents to all environments """
        
        for i,env in enumerate(self.values()):
            if i == 0: new_agents = env.add_agents(*args,**kwargs)
            else: env.add_agents( new_agents )


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
        if 'steps' in parameters: self.stop_if = self.stop_if_steps
        
        self.type = type(self).__name__
        self.run_id = run_id
        self.scenario = scenario
        
        # Counters
        self.t = 0 
        self._id_counter = 0  
        
        # Recording
        self.log = {}
        self.measure_log = {}
        self.output = data_dict()
        self.output.log = {}
        self.output.log['name'] = self.type
        self.output.log['time_stamp'] = str(datetime.now())
        
        # List of non-variable keys
        self._int_keys = list(self.keys())
 
    def _get_id(self):
        
        self._id_counter += 1 
        return self._id_counter - 1   
    
    def __repr__(self):
        
        rep = "Agent-based model {"
        ignore = ['model','log','measure_log','stop_if','key','output']
        
        for k,v in self.items():
            
            if k not in ignore and not k[0] == '_':
                rep += f"\n'{k}': {v}"
            
            if k == 'output':
                rep += f"\n'{k}': data_dict with {len(v.keys())} entries"

        return rep + ' }'
    
    def __getattr__(self, name): 
            
        try: # Access environments
            return self.envs[name]  
        except KeyError:
            raise AttributeError(f"Model '{self.type}' has no attribute or environment '{name}'") 
    
    # Creation & Destruction

    def add_env(self, env_key, env_class=environment, **kwargs):

        """ Creates a new environment """
        
        for env_key in make_list(env_key):
            self.envs[env_key] = env_class(self.model, env_key, **kwargs) 
                    
    def add_network(self, env_key, graph = None, env_class=network , **kwargs):
        
        """ Creates a new network environment """
        
        for env_key in make_list(env_key):
            self.add_env(env_key, env_class=env_class, graph=graph, **kwargs)
            
    def add_grid(self, env_key, env_class=grid , **kwargs):
        
        """ Creates a new spacial grid environment """
        
        for env_key in make_list(env_key):
            self.add_env(env_key, env_class=env_class, **kwargs)

    
    # Recording functions
            
    def measure(self, measure_key, value):
        
        """ Records an evaluation measure """
        
        self.measure_log[measure_key] = [value]
    
    def record_all(self):
        
        """ Records all dynamic model variables """
        
        keys = list(self.keys())
        for key in self._int_keys:
            keys.remove(key)
        
        self.record(keys)
    
    # Main simulation functions 
    
    def step(self):
        """ Defines the model's actions during each simulation step.
        Can be overwritten and used to perform the models' main dynamics."""
        pass
    
    def update(self):
        """ Defines the model's actions after setup and each simulation step.
        Can be overwritten and used for the recording of dynamic variables. """
        pass  
    
    def end(self):
        """ Defines the model's actions after the last simulation step.
        Can be overwritten and used to calculate evaluation measures."""
        pass
    
    def stop_if(self):
        """ 
        Stops :meth:`model.run` during an active simulation if it returns `True`. 
        Can be overwritten with a custom function.
        """
        return False
    
    def stop_if_steps(self):
        """ 
        Returns:
            bool: Whether time-step `t` has reached the parameter `model.p.steps`.
        """
        return self.t >= self.p.steps 
    
    def stop(self):
        """ Stops :meth:`model.run` during an active simulation. """
        self._stop = True       
            
    def run(self,display=True):
        
        """ Executes the simulation of the model.
        The order of events is as follows.
        The simulation starts at `t=0` and calls setup() and update().
        While stop_if() returns False and stop() hasn't been called,
        the simulation repeatedly calls t+=1, step(), and update().
        After the last step, end() is called.
        
        Arguments:
            display(bool,optional): Whether to display simulation progress (default True).
            
        Returns:
            data_dict: Recorded model data, also stored in `model.output`.
        """
        
        t0 = datetime.now() # Time-Stamp
        
        self._stop = False
        self.setup() 
        self.update()
        
        while not self.stop_if() and not self._stop:
            
            self.t += 1
            self.step()
            self.update()
            
            if display: print( f"\rCompleted: {self.t} steps" , end='' ) 
        
        self.end() 
        self.create_output()
        self.output.log['run_time'] = ct = str( datetime.now() - t0 )
        self.output.log['steps'] = self.t         
        
        if display: print(f"\nRun time: {ct}\nSimulation finished")
        
        return self.output
    
    def create_output(self):
        
        """ Generates an 'output' dictionary out of object logs """
    
        def output_from_obj_list(self,obj_list,c1,c2,columns):
            
            # Aggregate logs per object type
            obj_types = {}
            for obj in obj_list:
                
                if obj.log: # Check for variables
                    
                    # Add id to object log
                    obj.log['obj_id'] = [obj[c2]] * len(obj.log['t'])
                    
                    # Initiate object type if new
                    if obj.type not in obj_types.keys():
                        obj_types[obj.type] = {}
                    
                    # Add object log to aggr. log
                    for k,v in obj.log.items():
                        if k not in obj_types[obj.type]:
                            obj_types[obj.type][k] = []
                        obj_types[obj.type][k].extend(v)
            
            # Transform logs into dataframes
            for obj_type, log in obj_types.items():
                df = pd.DataFrame( log )
                for k,v in columns.items(): df[k]=v # Set additional index columns
                df = df.set_index(list(columns.keys())+['obj_id','t'])
                self.output['variables'][obj_type] =  df 
        
        # 0 - Document parameters
        if self.p: self.output['parameters'] = self.p
        
        # 1 - Define additional index columns 
        columns = {}
        if self.run_id is not None: columns['run_id'] = self.run_id
        if self.scenario is not None: columns['scenario'] = self.scenario
        
        # 2 - Create measure output
        if self.measure_log:   
            d = self.measure_log
            for key,value in columns.items(): d[key] = value
            df = pd.DataFrame(d)
            if columns: df = df.set_index(list(columns.keys()))
            self.output['measures'] = df
        
        # 3 - Create variable output
        self.output['variables'] = data_dict()
        
        # Create variable output for objects
        output_from_obj_list(self, self.agents, 'agent', 'id', columns)
        output_from_obj_list(self, self.envs.values(), 'env', 'key', columns)
        
        # Create variable output for model
        if self.log: 

            df = pd.DataFrame(self.log)
            df['obj_id'] = 'model'
            for k,v in columns.items(): df[k]=v
            df = df.set_index(list(columns.keys())+['obj_id','t'])

            if self.output['variables']: self.output['variables']['model'] = df
            else: self.output['variables'] = df # No subdict if only model vars
        
        # Remove variable dict if empty
        elif not self.output['variables']:
            del self.output['variables']

        
