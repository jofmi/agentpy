import pytest
import networkx as nx
import agentpy as ap


def test_add_agents():

    # Map agents to existing nodes

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
    model2.add_agents(2)
    model2.add_network()
    model2.env.add_agents(model2.agents)

    model2.env.graph.add_edge(model2.agents[0],model2.agents[1])

    assert model.env.graph.nodes.__repr__() == model2.env.graph.nodes.__repr__()
    assert model.env.graph.edges.__repr__() == model2.env.graph.edges.__repr__()