import pytest
import networkx as nx
import agentpy as ap


def test_add_agents():

    # Add agents to existing nodes
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    model = ap.Model()

    env = ap.Network(model, graph=graph)
    agents = ap.AgentList(model, 2)
    env.add_agents(agents, positions=env.nodes)
    for agent in agents:
        agent.pos = env.positions[agent]
    agents.node = env.nodes
    env.graph.add_edge(*agents.pos)

    # Test structure
    assert list(agents.pos) == list(agents.node)
    assert env.nodes == env.graph.nodes()
    assert list(env.graph.edges) == [tuple(agents.pos)]
    assert list(env.neighbors(agents[0]).id) == [3]

    # Add agents as new nodes
    model2 = ap.Model()
    agents2 = ap.AgentList(model2, 2)
    env2 = ap.Network(model2)
    env2.add_agents(agents2)
    for agent in agents2:
        agent.pos = env2.positions[agent]
    env2.graph.add_edge(*agents2.pos)

    # Test if the two graphs are identical
    assert env.graph.nodes.__repr__() == env2.graph.nodes.__repr__()
    assert env.graph.edges.__repr__() == env2.graph.edges.__repr__()


def test_move_agent():

    # Move agent one node to another
    model = ap.Model()
    graph = ap.Network(model)
    n1 = graph.add_node()
    n2 = graph.add_node()
    a = ap.Agent(model)
    graph.add_agents([a], positions=[n1])

    assert len(n1) == 1
    assert len(n2) == 0
    assert graph.positions[a] is n1

    graph.move_to(a, n2)

    assert len(n1) == 0
    assert len(n2) == 1
    assert graph.positions[a] is n2


def test_remove_agents():

    model = ap.Model()
    agents = ap.AgentList(model, 2)
    nw = ap.Network(model)
    nw.add_agents(agents)
    agent = agents[0]
    node = nw.positions[agent]
    nw.remove_agents(agent)
    assert len(nw.agents) == 1
    assert len(nw.nodes) == 2
    nw.remove_node(node)
    assert len(nw.agents) == 1
    assert len(nw.nodes) == 1
    agent2 = agents[1]
    nw.remove_node(nw.positions[agent2])
    assert len(nw.agents) == 0
    assert len(nw.nodes) == 0
