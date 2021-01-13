import pytest
import networkx as nx
import agentpy as ap


def test_add_agents():

    # Add agents to existing nodes
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    model = ap.Model()
    model.add_agents(2)
    model.add_network(graph=graph, agents=model.agents)
    model.env.graph.add_edge(model.agents[0],model.agents[1])
    assert list(model.agents[0].neighbors().id) == [2]

    # Add agents as new nodes
    model2 = ap.Model()
    agents = model2.add_agents(2)
    model2.add_network(agents = agents[0])  # Add at initialization
    model2.env.add_agents(agents[1])  # Add later
    model2.env.graph.add_edge(model2.agents[0],model2.agents[1])

    # Test if the two graphs are identical
    assert model.env.graph.nodes.__repr__() == model2.env.graph.nodes.__repr__()
    assert model.env.graph.edges.__repr__() == model2.env.graph.edges.__repr__()

    # Test errors
    model3 = ap.Model()
    graph = nx.Graph()
    model3.add_agents()
    with pytest.raises(ValueError):
        assert model3.add_network(graph=graph, agents=model3.agents)
    with pytest.raises(TypeError):
        assert model3.add_network(graph=1)


def test_remove_agents():

    model = ap.Model()
    model.add_agents(2)
    nw = model.add_network()
    nw.add_agents(model.agents)
    agent = model.agents[0]
    nw.remove_agents(agent)
    len(nw.agents) == 1
    len(nw.graph.nodes) == 1