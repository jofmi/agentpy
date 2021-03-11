import pytest
import agentpy as ap
import numpy as np
import scipy


def test_general():

    model = ap.Model()
    space = model.add_space((2, 2))
    assert space is model.env
    assert space.shape == (2, 2)
    assert space.dim == 2


def test_KDTree():

    model = ap.Model()
    space = model.add_space((2, 2))
    assert space.KDTree is None
    assert len(space.get_agents((1, 1), 2)) == 0
    space.add_agents(1)
    assert isinstance(space.KDTree, scipy.spatial.cKDTree)
    assert len(space.get_agents((1, 1), 2)) == 1


def test_add_agents_random():
    model = ap.Model()
    model.run(steps=0, seed=1, display=False)
    space = model.add_space((2, 2))
    agent = space.add_agents(1, random=True)[0]
    assert list(agent.position()) == [1.0236432494005134, 1.9009273926518706]


def test_remove():
    model = ap.Model()
    space = model.add_space((2, 2))
    space.add_agents(2)
    agent = model.agents[0]
    space.remove_agents(agent)
    assert len(space._agent_dict) == 1
    assert len(space.agents) == 1
    assert space.agents[0].id == 3


def test_positions():
    # Disconnected space
    model = ap.Model()
    space = model.add_space((2, 2))
    space.add_agents(1)
    space.add_agents(1, positions=[(1, 2)])

    # Position reference
    a1 = model.agents[0]
    a2 = model.agents[1]
    assert list(a1.position()) == list(space.position(a1))
    assert list(a1.position()) == list(space.position(a1.id))
    assert list(a1.position()) == [0, 0]
    assert list(a2.position()) == [1, 2]
    assert [list(x) for x in space.positions()] == [[0, 0], [1, 2]]
    assert list(space.positions(transpose=True)) == [[0, 1], [0, 2]]

    # Get agents
    assert len(space.get_agents((1, 1), 0.9)) == 0
    assert len(space.get_agents((1, 1), 1)) == 1
    assert len(space.get_agents((1, 1), 1.4)) == 1
    assert len(space.get_agents((1, 1), 1.42)) == 2

    # Get neighbors
    assert len(a1.neighbors(distance=2.0)) == 0
    assert len(a1.neighbors(distance=2.5)) == 1
    assert a1.neighbors(distance=2.5)[0] is a2

    # Movement restricted by border
    a2.move_by((2, -3))
    assert list(a2.position()) == [2, 0]

    # Move directly
    space.move_agent(a2.id, (1, 1))
    assert list(a2.position()) == [1, 1]

    # Connected space
    model = ap.Model()
    space = model.add_space((2, 2), connect_borders=True)
    space.add_agents(1)
    space.add_agents(1, positions=[(0, 1.9)])
    a1 = model.agents[0]
    a2 = model.agents[1]
    assert a1.neighbors(distance=0.11)[0] == a2

    # Movement over border
    a2.move_by((-3, 1.1))
    assert list(a2.position()) == [1, 1]
