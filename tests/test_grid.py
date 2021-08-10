import pytest
import agentpy as ap
import numpy as np
from agentpy.tools import AgentpyError


def make_grid(s, n=0, track_empty=False, agent_cls=ap.Agent):
    model = ap.Model()
    agents = ap.AgentList(model, n, agent_cls)
    grid = ap.Grid(model, (s, s), track_empty=track_empty)
    grid.add_agents(agents)
    return model, grid, agents


def test_general():
    model, grid, agents = make_grid(2)
    assert grid.shape == (2, 2)
    assert grid.ndim == 2


def test_add_agents():
    model = ap.Model()
    grid = ap.Grid(model, (2, 2))
    agents = ap.AgentList(model, 5)
    grid.add_agents(agents)
    assert grid.apply(len).tolist() == [[2, 1], [1, 1]]

    # Passed positions
    model = ap.Model()
    grid = ap.Grid(model, (2, 2))
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, [[0, 0], [1, 1]])
    assert grid.apply(len).tolist() == [[1, 0], [0, 1]]

    model = ap.Model()
    model.sim_setup(seed=1)
    grid = ap.Grid(model, (2, 2))
    agents = ap.AgentList(model, 5)
    grid.add_agents(agents, random=True)
    assert grid.apply(len).tolist() == [[0, 3], [1, 1]]

    with pytest.raises(AgentpyError):
        # Can't add more agents than empty positions
        model = ap.Model()
        model.sim_setup(seed=1)
        grid = ap.Grid(model, (2, 2), track_empty=True)
        agents = ap.AgentList(model, 5)
        grid.add_agents(agents, empty=True)

    with pytest.raises(AgentpyError):
        # Can't use empty if track_empty is False
        model = ap.Model()
        model.sim_setup(seed=1)
        grid = ap.Grid(model, (2, 2))
        agents = ap.AgentList(model, 5)
        grid.add_agents(agents, empty=True)

    model = ap.Model()
    model.sim_setup(seed=1)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, empty=True)
    assert grid.apply(len).tolist() == [[1, 1], [1, 1]]

    model = ap.Model()
    model.sim_setup(seed=1)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents)
    assert grid.apply(len).tolist() == [[2, 2], [0, 0]]

    model = ap.Model()
    model.sim_setup(seed=2)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, empty=True)
    agents = ap.AgentList(model, 1)
    grid.add_agents(agents, random=True, empty=True)
    assert grid.apply(len).tolist() == [[1, 1], [0, 1]]

    model = ap.Model()
    model.sim_setup(seed=2)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, empty=True)
    agents = ap.AgentList(model, 1)
    grid.add_agents(agents, random=True)
    assert grid.apply(len).tolist() == [[2, 1], [0, 0]]


def test_remove():
    model = ap.Model()
    agents = ap.AgentList(model, 2)
    grid = ap.Grid(model, (2, 2))
    grid.add_agents(agents)
    grid.remove_agents(agents[0])
    assert grid.apply(len).tolist() == [[0, 1], [0, 0]]

    # With track_empty
    model = ap.Model()
    agents = ap.AgentList(model, 2)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    grid.add_agents(agents)
    assert list(grid.empty) == [(1, 1), (1, 0)]
    grid.remove_agents(agents[0])
    assert list(grid.empty) == [(1, 1), (1, 0), (0, 0)]


def test_grid_iter():
    model = ap.Model()
    agents = ap.AgentList(model, 4)
    grid = ap.Grid(model, (2, 2))
    grid.add_agents(agents)
    assert len(grid.agents) == 4
    assert len(grid.agents[0:1, 0:1]) == 1


def test_attr_grid():
    model, grid, agents = make_grid(2, 4)
    assert grid.attr_grid('id').tolist() == [[1, 2], [3, 4]]


def test_apply():
    model, grid, agents = make_grid(2, 4)
    assert grid.apply(len).tolist() == [[1, 1], [1, 1]]


def test_move():
    model, grid, agents = make_grid(2, 2, track_empty=True)
    agent = agents[0]
    assert grid.attr_grid('id').tolist()[0] == [1., 2.]
    grid.move_to(agent, (1, 0))  # Move in absolute terms
    grid.move_to(agent, (1, 0))  # Moving to same pos causes no error
    assert grid.attr_grid('id').tolist()[0][1] == 2.0
    assert grid.attr_grid('id').tolist()[1][0] == 1.0
    assert np.isnan(grid.attr_grid('id').tolist()[1][1])
    assert list(grid.empty) == [(1, 1), (0, 0)]
    grid.move_by(agent, (-1, 0))  # Move in relative terms
    assert grid.attr_grid('id').tolist()[0] == [1., 2.]
    assert list(grid.empty) == [(1, 1), (1, 0)]


def test_move_empty_multiple_agents():
    model = ap.Model()
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 3)
    agent = agents[0]
    grid.add_agents(agents, [(0, 0), (0, 0), (0, 1)])
    assert list(grid.empty) == [(1, 1), (1, 0)]
    grid.move_to(agent, (1, 1))
    assert list(grid.empty) == [(1, 0)]
    grid.move_to(agent, (0, 0))
    assert list(grid.empty) == [(1, 0), (1, 1)]
    grid.move_to(agent, (0, 1))
    assert list(grid.empty) == [(1, 0), (1, 1)]


def test_move_torus():
    model = ap.Model()
    agents = ap.AgentList(model, 1)
    agent, = agents
    grid = ap.Grid(model, (4, 4), torus=True)
    grid.add_agents(agents, [[0, 0]])

    assert grid.positions[agent] == (0, 0)
    grid.move_by(agent, [-1, -1])
    assert grid.positions[agent] == (3, 3)
    grid.move_by(agent, [1, 0])
    assert grid.positions[agent] == (0, 3)
    grid.move_by(agent, [0, 1])
    assert grid.positions[agent] == (0, 0)

    model = ap.Model()
    agents = ap.AgentList(model, 1)
    agent, = agents
    grid = ap.Grid(model, (4, 4), torus=False)
    grid.add_agents(agents, [[0, 0]])

    assert grid.positions[agent] == (0, 0)
    grid.move_by(agent, [-1, -1])
    assert grid.positions[agent] == (0, 0)
    grid.move_by(agent, [6, 6])
    assert grid.positions[agent] == (3, 3)


def test_neighbors():
    model, grid, agents = make_grid(5, 25)
    a = agents[12]
    assert list(grid.neighbors(a)) == list(grid.neighbors(a))
    assert len(grid.neighbors(a, distance=1)) == 8
    assert len(grid.neighbors(a, distance=2)) == 24


def test_neighbors_with_torus():

    model = ap.Model()
    agents = ap.AgentList(model, 5)
    grid = ap.Grid(model, (4, 4), torus=True)
    grid.add_agents(agents, [[0, 0], [1, 3], [2, 0], [3, 2], [3, 3]])

    grid.apply(len).tolist()

    assert list(grid.neighbors(agents[0]).id) == [5,2]

    model = ap.Model()
    agents = ap.AgentList(model, 5)
    grid = ap.Grid(model, (4, 4), torus=True)
    grid.add_agents(agents, [[0, 1], [1, 3], [2, 0], [3, 2], [3, 3]])

    grid.apply(len).tolist()

    assert list(grid.neighbors(agents[0]).id) == [4]
    assert list(grid.neighbors(agents[1]).id) == [3]

    for d in [2, 3, 4]:

        model = ap.Model()
        agents = ap.AgentList(model, 5)
        grid = ap.Grid(model, (4, 4), torus=True)
        grid.add_agents(agents, [[0, 1], [1, 3], [2, 0], [3, 2], [3, 3]])

        grid.apply(len).tolist()

        assert list(grid.neighbors(agents[0], distance=d).id) == [2, 3, 4, 5]
        assert list(grid.neighbors(agents[1], distance=d).id) == [1, 3, 4, 5]


def test_field():
    model = ap.Model()
    grid = ap.Grid(model, (2, 2))

    grid.add_field('f1', np.array([[1, 2], [3, 4]]))
    grid.add_field('f2', 5)

    assert grid.f1.tolist() == [[1, 2], [3, 4]]

    grid.f1[1, 1] = 8

    assert grid.f1.tolist() == [[1, 2], [3, 8]]

    assert grid.f2.tolist() == [[5, 5], [5, 5]]
    assert grid.grid.f2.tolist() == grid.f2.tolist()

    grid.del_field('f2')

    with pytest.raises(AttributeError):
        grid.f2

    with pytest.raises(AttributeError):
        grid.grid.f2