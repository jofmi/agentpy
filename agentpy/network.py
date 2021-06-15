""" Agentpy Network Module """

import itertools
import networkx as nx
from .objects import Object
from .sequences import AgentList, AgentIter, AttrIter
from .tools import make_list


class AgentNode(set):
    """ Node of :class:`Network`. Functions like a set of agents. """

    # TODO Connector between AgentNode attributes and the networkx attr dict

    def __init__(self, label):
        self.label = label

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"AgentNode ({self.label})"


class Network(Object):
    """ Agent environment with a graph topology.
    Every node of the network is a :class:`AgentNode` that can hold
    multiple agents as well as node attributes.

    This class can be used as a parent class for custom network types.
    All agentpy model objects call the method :func:`setup` after creation,
    and can access class attributes like dictionary items.

    Arguments:
        model (Model): The model instance.
        graph (networkx.Graph, optional): The environments' graph.
            Can also be a DiGraph, MultiGraph, or MultiDiGraph.
            Nodes will be converted to :class:`AgentNode`,
            with their original label being kept as `AgentNode.label`.
            If none is passed, an empty :class:`networkx.Graph` is created.
        **kwargs: Will be forwarded to :func:`Network.setup`.

    Attributes:
        graph (networkx.Graph): The network's graph instance.
        agents (AgentIter): Iterator over the network's agents.
        nodes (AttrIter): Iterator over the network's nodes.
    """

    def __init__(self, model, graph=None, **kwargs):

        super().__init__(model)
        self._i = -1  # Node label counter
        self.positions = {}  # Agent Instance : Node reference

        if graph is None:
            self.graph = nx.Graph()
        else:
            nodes = graph.nodes
            self._i = len(nodes)
            mapping = {i: AgentNode(label=i) for i in nodes}
            self.graph = nx.relabel_nodes(graph, mapping=mapping)

        self._set_var_ignore()
        self.setup(**kwargs)

    @property
    def agents(self):
        return AgentIter(self.model, self.positions.keys())

    @property
    def nodes(self):
        return AttrIter(self.graph.nodes)

    # Add and remove nodes -------------------------------------------------- #

    def add_node(self, label=None):
        """ Adds a new node to the network.

        Arguments:
            label (int or string, optional): Unique name of the node,
                which must be different from all other nodes.
                If none is passed, an integer number will be chosen.

        Returns:
            AgentNode: The newly created node.
        """
        self._i += 1
        if label is None:
            label = self._i
        node = AgentNode(label=label)
        self.graph.add_node(node)
        return node

    def remove_node(self, node):
        """ Removes a node from the network.

        Arguments:
            node (AgentNode): Node to be removed.
        """
        self.remove_agents(node)
        self.graph.remove_node(node)

    # Add and remove agents ------------------------------------------------- #

    def add_agents(self, agents, positions=None):
        """ Adds agents to the network environment.

        Arguments:
            agents (Sequence of Agent):
                Instance or iterable of agents to be added.
            positions (Sequence of AgentNode, optional):
                The positions of the agents.
                Must have the same length as 'agents',
                with each entry being an :class:`AgentNode` of the network.
                If none is passed, new nodes will be created for each agent.
        """

        if positions is None:
            for agent in agents:
                node = self.add_node()
                node.add(agent)
                self.positions[agent] = node
        else:
            for agent, node in zip(agents, positions):
                node.add(agent)
                self.positions[agent] = node

    def remove_agents(self, agents):
        """ Removes agents from the network. """
        for agent in make_list(agents):
            self.positions[agent].remove(agent)
            del self.positions[agent]

    # Move and select agents ------------------------------------------------ #

    def move_to(self, agent, node):
        """ Moves agent to new position.

        Arguments:
            agent (Agent): Instance of the agent.
            node (AgentNode): New position of the agent.
        """

        node.add(agent)
        self.positions[agent].remove(agent)
        self.positions[agent] = node

    def neighbors(self, agent):
        """ Select agents from neighboring nodes.
        Does not include other agents from the agents' own node.

        Arguments:
            agent (Agent): Instance of the agent.

        Returns:
            AgentIter: Iterator over the selected neighbors.
        """

        # TODO Improve
        nodes = self.graph.neighbors(self.positions[agent])
        return AgentIter(self.model, itertools.chain.from_iterable(nodes))
