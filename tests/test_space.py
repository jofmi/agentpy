import pytest
import agentpy as ap
import numpy as np
import scipy


def make_space(s, n=0, torus=False):

    model = ap.Model()
    agents = ap.AgentList(model, n)
    space = ap.Space(model, (s, s), torus=torus)
    space.add_agents(agents)
    for agent in agents:
        agent.pos = space.positions[agent]
    return model, space, agents


def test_general():

    model, space, agents = make_space(2)
    assert space.shape == (2, 2)
    assert space.ndim == 2


def test_KDTree():

    model, space, agents = make_space(2)
    assert space.kdtree is None
    assert len(space.select((1, 1), 2)) == 0
    space.add_agents([ap.Agent(model)])
    assert isinstance(space.kdtree, scipy.spatial.cKDTree)
    assert len(space.select((1, 1), 2)) == 1


def test_add_agents_random():
    model, space, agents = make_space(2)
    model.run(steps=0, seed=1, display=False)
    agent = ap.Agent(model)
    space.add_agents([agent], random=True)
    assert list(space.positions[agent]) == [1.527549237953228, 0.5101380514788434]


def test_remove():
    model, space, agents = make_space(2, 2)
    agent = agents[0]
    space.remove_agents(agent)
    assert len(space.positions) == 1
    assert len(space.agents) == 1
    assert list(space.agents)[0].id == 2


def test_positions():

    # Disconnected space
    model, space, agents = make_space(2)
    a1 = ap.Agent(model)
    a2 = ap.Agent(model)
    space.add_agents([a1])
    space.add_agents([a2], positions=[(1, 2)])

    # Position reference
    assert list(space.positions[a1]) == [0, 0]
    assert list(space.positions[a2]) == [1, 2]
    assert [list(x) for x in space.positions.values()] == [[0, 0], [1, 2]]

    # Get agents
    assert len(space.select((1, 1), 0.9)) == 0
    assert len(space.select((1, 1), 1)) == 1
    assert len(space.select((1, 1), 1.4)) == 1
    assert len(space.select((1, 1), 1.42)) == 2

    # Get neighbors
    assert len(space.neighbors(a1, distance=2.0)) == 0
    assert len(space.neighbors(a1, distance=2.5)) == 1
    assert list(space.neighbors(a1, distance=2.5))[0] is a2

    # Movement restricted by border
    space.move_by(a2, (2, -3))
    assert list(space.positions[a2]) == [2, 0]

    # Move directly
    space.move_to(a2, (1, 1))
    assert list(space.positions[a2]) == [1, 1]

    # Connected space (toroidal)
    model, space, agents = make_space(2, torus=True)
    a1 = ap.Agent(model)
    a2 = ap.Agent(model)
    space.add_agents([a1])
    space.add_agents([a2], positions=[(0, 1.9)])
    assert list(space.neighbors(a1, distance=0.11))[0] == a2

    # Movement over border
    space.move_by(a2, (-3, 1.1))
    assert list(space.positions[a2]) == [1, 1]
