import pytest
import agentpy as ap
import numpy as np
from agentpy.tools import AgentpyError


def make_grid(s, n=0):
    model = ap.Model()
    agents = ap.AgentList(model, n)
    grid = ap.Grid(model, (s, s))
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

    model = ap.Model()
    model.run_setup(seed=1)
    grid = ap.Grid(model, (2, 2))
    agents = ap.AgentList(model, 5)
    grid.add_agents(agents, random=True)
    assert grid.apply(len).tolist() == [[0, 3], [1, 1]]

    with pytest.raises(AgentpyError):
        # Can't add more agents than empty positions
        model = ap.Model()
        model.run_setup(seed=1)
        grid = ap.Grid(model, (2, 2), track_empty=True)
        agents = ap.AgentList(model, 5)
        grid.add_agents(agents, empty=True)

    with pytest.raises(AgentpyError):
        # Can't use empty if track_empty is False
        model = ap.Model()
        model.run_setup(seed=1)
        grid = ap.Grid(model, (2, 2))
        agents = ap.AgentList(model, 5)
        grid.add_agents(agents, empty=True)

    model = ap.Model()
    model.run_setup(seed=1)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, empty=True)
    assert grid.apply(len).tolist() == [[1, 1], [1, 1]]

    model = ap.Model()
    model.run_setup(seed=1)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents)
    assert grid.apply(len).tolist() == [[2, 2], [0, 0]]

    model = ap.Model()
    model.run_setup(seed=2)
    grid = ap.Grid(model, (2, 2), track_empty=True)
    agents = ap.AgentList(model, 2)
    grid.add_agents(agents, empty=True)
    agents = ap.AgentList(model, 1)
    grid.add_agents(agents, random=True, empty=True)
    assert grid.apply(len).tolist() == [[1, 1], [0, 1]]

    model = ap.Model()
    model.run_setup(seed=2)
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


def test_movement():
    model, grid, agents = make_grid(2, 2)
    agent = agents[0]
    assert grid.attr_grid('id').tolist()[0] == [1., 2.]
    agent.move_to((1, 0))  # Move in absolute terms
    assert grid.attr_grid('id').tolist()[0][1] == 2.0
    assert grid.attr_grid('id').tolist()[1][0] == 1.0
    assert np.isnan(grid.attr_grid('id').tolist()[1][1])
    agent.move_by((-1, 0))  # Move in relative terms
    assert grid.attr_grid('id').tolist()[0] == [1., 2.]


def test_neighbors():
    model, grid, agents = make_grid(5, 25)
    a = agents[12]
    assert list(a.neighbors()) == list(grid.neighbors(a))
    assert len(a.neighbors(distance=1)) == 8
    assert len(a.neighbors(distance=2)) == 24
