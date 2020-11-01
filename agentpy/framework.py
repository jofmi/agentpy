""" Main framework for agent-based models """

import pandas as pd
import networkx as nx
import random as rd
import operator

from datetime import datetime
from itertools import product

from .output import DataDict
from .tools import make_list, make_matrix, AttrDict, ObjList, AgentpyError


# (!) Change into parents class
def record(self, var_keys, value=None):
    """ Records an objects variables. 
    
    Arguments:
        var_keys(str or list): Names of the variables
        value(optional): Value to be recorded.
            If no value is given, var_keys are looked up as attributes.
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

        if value is None:
            v = self[var_key]
        else:
            v = value

        self.log[var_key][-1] = v


def setup(self):
    """Can be overwritten to define the 
    object's actions after creation."""
    pass


# Level 3 - Agent class
#######################   

class Agent(AttrDict):
    """ Individual agent of an agent-based model.

    This class can be used as a parent class for custom agent types.
    :meth:`Agent.setup` will be called automatically after agent creation.  
    Attributes can be accessed as items (see :class:`AttrDict`).
    
    Attributes:
        model (Model): The agents' model 
        envs (EnvDict): The agents' environments
        p (AttrDict): The models' parameters
        log (dict): The agents' recorded variables
        type (str): The agents' class name
        t0 (int): Time-step of the agent's creation
        id (int): The agent's unique identifier
        
    Arguments:
        model (Model): The agents' model 
        envs (dict or EnvDict, optional): The agents' environments
    
    """

    setup = setup
    record = record

    def __init__(self, model, envs=None):

        super().__init__()

        self.model = model
        self.p = model.p
        self.t0 = model.t
        self.id = model._get_id()

        self.envs = EnvDict(model)
        if envs: self.envs.update(envs)

        self.log = {}
        self.type = type(self).__name__

        self.setup()  # Custom initialization

    def __repr__(self):
        s = f"Agent {self.id}"
        if self.type != 'Agent': s += f" ({self.type})"
        return s

    def __hash__(self):
        return self.id  # Necessary for networkx

    def pos(self, key=None):

        # Select network
        if key == None:
            for k, v in self.envs.items():
                if v.topology == 'grid':
                    key = k
                    break

        return self.envs[key].pos[self]

    def neighbors(self, env_keys=None):
        """ Returns the agents' neighbor's from grids and/or networks.
        
        Arguments:
            env_keys(str or list of str, optional): Names of environments 
                from the agent with type :class:`network` or :class:`grid`.
                If none are given, all such environments are selected.

        Returns: 
            AgentList
                
        """

        # Select environments
        if env_keys is None:
            keys = [k for k, v in self.envs.items()
                    if v.topology in ['network', 'grid']]

        # Select neighbors
        agents = AgentList()
        for key in make_list(env_keys):
            try:
                agents.append(self.envs[key].neighbors(self))
            except KeyError:
                AgentpyError(f"Agent {self.id} has no environment '{key}'")

        return agents

    def exit(self, env_keys=None):
        """ Removes the agent from environments and/or the model.
        
        Arguments:
            env_keys(str or list, optional): Environments from which the agent should be removed.
                If no keys are given, agent is removed from the model completely.
        """

        if env_keys is None:
            envs = self.envs  # Select all environments by default
            self.model.agents.remove(self)  # Delete from model
        else:
            envs = {k: v for k, v in self.envs.items() if k in env_keys}

        for key, env in envs.items():
            env.agents.remove(self)
            del self.envs[key]


class AgentList(ObjList):
    """ List of agents.
    
    Attribute calls and operators are applied to all agents
    and return a list of return values (see :class:`ObjList`).
    """

    # def __init__(self, agents = None): (!)
    #    super().__init__()
    #    if agents: self.extend( make_list(agents) )
    # Arguments:
    #         agents(Agent or list of Agent, optional): Initial agent entries

    def __repr__(self):
        return f"AgentList [{len(self)} agents]"

    def do(self, method, *args, shuffle=False, **kwargs):
        """ Calls ``method(*args,**kwargs)`` for all agents in the list.
        
        Arguments:
            method (str): Name of the method
            *args: Will be forwarded to the agents method
            shuffle (bool, optional): Whether to shuffle order 
                in which agents are called (default False)
            **kwargs: Will be forwarded to the agents method
        """
        if shuffle:
            agents = list(self)  # Soft Copy
            rd.shuffle(agents)
        else:
            agents = self

        for agent in agents:
            getattr(agent, method)(*args, **kwargs)

    def select(self, var_key, value=True, rel="=="):
        """ Returns a new :class:`AgentList` of selected agents.
        
        Arguments:
            var_key (str): Variable for selection.
            value (optional): Value for selection (default True).
            rel (str, optional): Relation between variable and value.
                Options: '=='(default), '!=', '<', '<=', '>', '>='.
        """

        relations = {'==': operator.eq, '!=': operator.ne,
                     '<': operator.lt, '<=': operator.le,
                     '>': operator.gt, '>=': operator.ge}

        return AgentList([a for a in self if relations[rel](a[var_key], value)])

    def assign(self, var_key, value):
        """ Assigns ``value`` to ``var_key`` for each agent."""
        for agent in self: setattr(agent, var_key, value)

    def of_type(self, agent_type):
        """ Returns a new :class:`AgentList` with agents of type ``agent_type``. """
        return self.select('type', agent_type)

    def random(self, n=1):
        """ Returns a new :class:`AgentList` with ``n`` random agents (default 1)."""
        return AgentList(rd.sample(self, n))


### Level 2 - Environment class ###        

def add_agents(self, agents, agent_class=Agent, **kwargs):
    """ Adds agents to the environment.
    
    Arguments:
        agents(int or AgentList): Number of new agents or list of existing agents.
        agent_class(class): Type of agent to be created if int is passed for agents.
    """

    if isinstance(agents, int):
        # Create new agent instances
        agents = AgentList([agent_class(self.model, {self.key: self}, **kwargs)
                            for _ in range(agents)])
        # Add agents to master list
        if self != self.model:
            self.model.agents.extend(agents)

    else:
        # Add existing agents
        if not isinstance(agents, AgentList):
            agents = AgentList(agents)

        # Add new environment to agents
        if self != self.model:
            for agent in agents:
                agent.envs[self.key] = self

    # Add agents to this env
    self.agents.extend(agents)

    return agents


def _env_init(self, model, key):
    AttrDict.__init__(self)
    self.model = model
    self.agents = AgentList()
    self.envs = EnvDict(model)
    self.p = model.p
    self.key = key
    self.type = type(self).__name__
    self.t0 = model.t
    self.log = {}


class Environment(AttrDict):
    """ Standard environment for agents.

    This class can be used as a parent class for custom environment types. 
    Attributes can be accessed as items (see :class:`AttrDict`).
    
    Attributes:
        model (Model): The model instance
        agents (AgentList): The environments' agents
        p (AttrDict): The models' parameters
        type (str): The environments' type 
        key(str): The environments' name
        topology(str): Topology of the environment
        t0 (int): Time-step of the environments' creation
        log (dict): The environments' recorded variables
        
    Arguments:
        model (Model): The environments' model 
        key (str, optional): The environments' name
    
    """

    setup = setup
    add_agents = add_agents
    record = record

    def __init__(self, model, key):
        _env_init(self, model, key)
        self.topology = None
        self.setup()

    def __repr__(self):
        rep = f"Environment '{self.key}'"
        if type != "Environment":
            rep += f" of type '{self.type}'"
        return rep  # (!) try with alias!

    def neighbors(self):
        raise AgentpyError(f"Environment {self.key} has no topology for neighbors.")


class Network(Environment):
    """ Agent environment with a graph topology.

    Unknown attribute and method calls are forwarded to self.graph.
    Inherits attributes and methods from :class:`environment`.
    
    Attributes:
        graph(networkx.Graph): Networkx Graph instance
    
    Arguments:
        model (Model): The model instance
        key (str, optional): The environments' name
        graph(networkx.Graph): The environments' graph
    """

    def __init__(self, model, env_key, graph=None):

        _env_init(self, model, env_key)

        if graph is None:
            self.graph = nx.Graph()
        elif isinstance(graph, nx.Graph):
            self.graph = graph
        else:
            raise TypeError("Argument 'graph' must be of type networkx.Graph")

        self.topology = 'network'
        self.setup()

    def add_agents(self, agents, agent_class=Agent, map_to_nodes=False, **kwargs):
        """ Adds agents to the network.

        Arguments:
            agents(int, Agent, list, or AgentList): Number of new agents or existing agents.
            agent_class(class): Type of agent to be created if int is passed for agents.
            map_to_nodes(bool,optional): Map new agents to each node of the graph (default False).
                Should be used if a graph with empty nodes has been passed at network creation.
        """

        # Standard adding
        new_agents = add_agents(self, agents, agent_class, **kwargs)

        # Extra network features
        if map_to_nodes:
            # Map each agent to a node of the graph
            if len(new_agents) != len(self.graph.nodes):
                raise AgentpyError(
                    f"Number of agents ({len(new_agents)}) does not match number of nodes ({len(self.graph.nodes)})")
            mapping = {i: agent for i, agent in enumerate(new_agents)}
            nx.relabel_nodes(self.graph, mapping=mapping, copy=False)
        else:
            # Add agents to graph as new nodes
            for agent in new_agents:
                self.graph.add_node(agent)

    def neighbors(self, agent):
        """ Returns :class:`AgentList` of agents that are connected to the passed agent. """
        return AgentList([n for n in self.graph.neighbors(agent)])

    def __getattr__(self, name):

        # Forward unknown method call to self.graph

        def method(*args, **kwargs):
            return getattr(self.graph, name)(*args, **kwargs)

        try:
            return method
        except AttributeError:
            raise AttributeError(f"module {__name__} has no attribute {name}")


class Grid(Environment):
    """ Grid environment that contains agents with a spatial topology.
    Inherits attributes and methods from :class:`Environment`.
    
    Attributes:
        pos(dict): Agent positions
    
    Arguments:
        model (Model): The model instance 
        key (dict or EnvDict, optional):  The environments' name
        dim(int, optional): Number of dimensions (default 2).
        size(int or tuple): Size of the grid.
            If int, the same length is assigned to each dimension.
            If tuple, one int item is required per dimension.
    """

    def __init__(self, model, env_key, shape):

        _env_init(self, model, env_key)

        self.topology = 'grid'
        self.grid = make_matrix(make_list(shape), AgentList)
        self.shape = shape
        self.pos = {}

        self.setup()

    def neighbors(self, agent, shape='diamond'):

        """ Return agent neighbors """

        if shape == 'diamond':
            return self._get_neighbors4(self.pos[agent], self.grid)
        elif shape == 'square':
            return self._get_neighbors8(self.pos[agent], self.grid)

    def area(self, area):

        return AgentList(self._get_area(area, self.grid))

    def _get_area(self, area, grid):

        """ Return agents in area of style: [(x_min,x_max),(y_min,y_max),...]"""

        subgrid = grid[area[0][0]:area[0][1] + 1]

        if isinstance(subgrid[0], AgentList):  # Detect last row (must have AgentLists)
            return [y for x in subgrid for y in x]  # Flatten list of AgentLists to list of agents

        objects = []
        for row in subgrid:
            objects.extend(self._get_area(area[1:], row))

        return objects

    def _get_neighbors4(self, pos, grid, dist=1):

        """ Return agents in diamond-shaped area around pos """

        subgrid = grid[max(0, pos[0] - dist):pos[0] + dist + 1]

        if len(pos) == 1:
            return [y for x in subgrid for y in x]  # flatten list

        objects = []
        for row, dist in zip(subgrid, [0, 1, 0]):
            objects.extend(self._get_neighbors4(pos[1:], row, dist))

        return objects

    def _get_neighbors8(self, pos, grid):

        subgrid = grid[max(0, pos[0] - 1):pos[0] + 2]

        if len(pos) == 1:
            return [y for x in subgrid for y in x]  # flatten list

        objects = []
        for row in subgrid:
            objects.extend(self._get_neighbors8(pos[1:], row))

        return objects

    def _get_pos(self, grid, pos):
        if len(pos) == 1: return grid[pos[0]]
        return self._get_pos(grid[pos[0]], pos[1:])

    def get_pos(self, pos):
        return self._get_pos(self.grid, pos)

    def change_pos(self, agent, new_position):
        self.get_pos(self.pos[agent], self.grid).drop(agent)  # Remove from old position
        self.get_pos(new_position, self.grid).append(agent)  # Add to new position
        self.pos[agent] = new_position  # Log Position

    def add_agents(self, agents, agent_class=Agent, positions=None,
                   random=False, map_to_grid=False, **kwargs):

        """ Adds agents to the grid environment."""

        # (!) unfinished

        # Standard adding
        new_agents = add_agents(self, agents, agent_class, **kwargs)

        # Extra grid features
        if map_to_grid:
            pass  # (!)
        elif positions:
            for agent, pos in zip(new_agents, positions):
                self.pos[agent] = pos
        elif random:
            positions = list(product(*[range(i) for i in self.shape]))
            sample = rd.sample(positions, agents)

            for agent, pos in zip(new_agents, sample):
                self.pos[agent] = pos  # Log Position
                self.get_pos(pos).append(agent)  # Add to new position

        else:
            for agent in new_agents:
                self.pos[agent] = [0] * self.dim


class EnvDict(dict):
    """ Dictionary for environments
    
    Attributes:
        model(model): The current model instance
    """

    def __init__(self, model):

        super().__init__()
        self.model = model

    def __repr__(self):

        return f"EnvDict {{{len(self.keys())} environments}}"

    def of_type(self, env_type):

        """ Returns :class:`EnvDict` of selected environments. """

        new_dict = EnvDict(self.model)
        selection = {k: v for k, v in self.items() if v.type == env_type}
        new_dict.update(selection)
        return new_dict

    def do(self, method, *args, **kwargs):

        """ Calls ``method(*args,**kwargs)`` for all environments in the dictionary. """

        for env in self.values(): getattr(env, method)(*args, **kwargs)

    def add_agents(self, *args, **kwargs):

        """ Calls :meth:`Environment.add_agents` with `*args,**kwargs` 
        and forwards new agents to all environments in the dictionary."""

        for i, env in enumerate(self.values()):
            if i == 0:
                new_agents = env.add_agents(*args, **kwargs)
            else:
                env.add_agents(new_agents)


### Level 1 - Model Class ###

class Model(AttrDict):
    """ An agent-based model.

    This class can be used as a parent class for custom models. 
    Attributes can be accessed as items (see :class:`AttrDict`).
    Environments can be called as attributes by their keys.
    
    Attributes:
        model(model): Reference to self
        envs(EnvDict): The models' environments
        agents(AgentList): The models' agents
        p(AttrDict): The models' parameters
        type(str): The models' name
        t (int): Current time-step of the model
        log (dict): The models' recorded variables
        output(DataDict): Simulation output data
        
    Arguments:
        parameters(dict,optional): Model parameters
        run_id(int,optional): Number of current run
        scenario(str,optional): Current scenario
    """

    setup = setup
    record = record
    add_agents = add_agents

    def __init__(self,
                 parameters=None,
                 run_id=None,
                 scenario=None
                 ):

        super().__init__()

        self.model = self
        self.key = 'model'
        self.envs = EnvDict(self)
        self.agents = AgentList()
        self.p = AttrDict()
        if parameters:
            self.p.update(parameters)
        if 'steps' in self.p:
            self.stop_if = self.stop_if_steps

        self.type = type(self).__name__
        self.run_id = run_id
        self.scenario = scenario

        # Counters
        self.t = 0
        self._id_counter = 0

        # Recording
        self.log = {}
        self.measure_log = {}
        self.output = DataDict()
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
        ignore = ['model', 'log', 'measure_log', 'stop_if', 'key', 'output']

        for k, v in self.items():

            if k not in ignore and not k[0] == '_':
                rep += f"\n'{k}': {v}"

            if k == 'output':
                rep += f"\n'{k}': DataDict with {len(v.keys())} entries"

        return rep + ' }'

    def __getattr__(self, name):

        try:  # Access environments
            return self.envs[name]
        except KeyError:
            raise AttributeError(f"Model '{self.type}' has no attribute or environment '{name}'")

            # Creation & Destruction

    def add_env(self, env_key, env_class=Environment, **kwargs):

        """ Creates a new environment """

        for env_key in make_list(env_key):
            self.envs[env_key] = env_class(self.model, env_key, **kwargs)

    def add_network(self, env_key, graph=None, env_class=Network, **kwargs):

        """ Creates a new network environment """

        for env_key in make_list(env_key):
            self.add_env(env_key, env_class=env_class, graph=graph, **kwargs)

    def add_grid(self, env_key, env_class=Grid, **kwargs):

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

    def run(self, display=True):

        """ Executes the simulation of the model.
        The order of events is as follows.
        The simulation starts at `t=0` and calls setup() and update().
        While stop_if() returns False and stop() hasn't been called,
        the simulation repeatedly calls t+=1, step(), and update().
        After the last step, end() is called.
        
        Arguments:
            display(bool,optional): Whether to display simulation progress (default True).
            
        Returns:
            DataDict: Recorded model data, also stored in `model.output`.
        """

        t0 = datetime.now()  # Time-Stamp

        self._stop = False
        self.setup()
        self.update()

        while not self.stop_if() and not self._stop:

            self.t += 1
            self.step()
            self.update()

            if display: print(f"\rCompleted: {self.t} steps", end='')

        self.end()
        self.create_output()
        self.output.log['run_time'] = ct = str(datetime.now() - t0)
        self.output.log['steps'] = self.t

        if display: print(f"\nRun time: {ct}\nSimulation finished")

        return self.output

    def create_output(self):

        """ Generates an 'output' dictionary out of object logs """

        def output_from_obj_list(self, obj_list, c1, c2, columns):

            # Aggregate logs per object type
            obj_types = {}
            for obj in obj_list:

                if obj.log:  # Check for variables

                    # Add id to object log
                    obj.log['obj_id'] = [obj[c2]] * len(obj.log['t'])

                    # Initiate object type if new
                    if obj.type not in obj_types.keys():
                        obj_types[obj.type] = {}

                    # Add object log to aggr. log
                    for k, v in obj.log.items():
                        if k not in obj_types[obj.type]:
                            obj_types[obj.type][k] = []
                        obj_types[obj.type][k].extend(v)

            # Transform logs into dataframes
            for obj_type, log in obj_types.items():
                df = pd.DataFrame(log)
                for k, v in columns.items(): df[k] = v  # Set additional index columns
                df = df.set_index(list(columns.keys()) + ['obj_id', 't'])
                self.output['variables'][obj_type] = df

                # 0 - Document parameters

        if self.p: self.output['parameters'] = self.p

        # 1 - Define additional index columns 
        columns = {}
        if self.run_id is not None: columns['run_id'] = self.run_id
        if self.scenario is not None: columns['scenario'] = self.scenario

        # 2 - Create measure output
        if self.measure_log:
            d = self.measure_log
            for key, value in columns.items(): d[key] = value
            df = pd.DataFrame(d)
            if columns: df = df.set_index(list(columns.keys()))
            self.output['measures'] = df

        # 3 - Create variable output
        self.output['variables'] = DataDict()

        # Create variable output for objects
        output_from_obj_list(self, self.agents, 'agent', 'id', columns)
        output_from_obj_list(self, self.envs.values(), 'env', 'key', columns)

        # Create variable output for model
        if self.log:

            df = pd.DataFrame(self.log)
            df['obj_id'] = 'model'
            for k, v in columns.items(): df[k] = v
            df = df.set_index(list(columns.keys()) + ['obj_id', 't'])

            if self.output['variables']:
                self.output['variables']['model'] = df
            else:
                self.output['variables'] = df  # No subdict if only model vars

        # Remove variable dict if empty
        elif not self.output['variables']:
            del self.output['variables']
