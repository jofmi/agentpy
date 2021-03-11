import pytest
import agentpy as ap
import numpy as np


def test_general():

    model = ap.Model()
    grid = model.add_grid((2, 2))
    assert grid is model.env
    assert grid.shape == (2, 2)
    assert grid.dim == 2

    # 5th agent must be at first position again
    model.env.add_agents(5)
    assert len(model.env.grid[0][0]) == 2
    assert len(model.env.grid[0][1]) == 1

    # Test only for errors
    model.env.add_agents(5, random=True)


def test_remove():
    model = ap.Model()
    grid = model.add_grid((2, 2))
    grid.add_agents(2)
    agent = model.agents[0]
    grid.remove_agents(agent)
    assert grid.attribute('id') == [[np.nan, 3], [np.nan, np.nan]]
    assert len(grid._agent_dict) == 1
    assert len(grid.agents) == 1


def test_positions():
    model = ap.Model()
    grid = model.add_grid((2, 2))
    grid.add_agents(5)

    # Position reference
    a = model.agents[0]
    assert a.position() == (0, 0)
    assert a.position() is grid.position(a)  # By instance
    assert a.position() is grid.position(a.id)  # By id

    # Positions
    assert list(grid.positions()) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert list(grid.positions([(0, 0), (0, 1)])) == [(0, 0), (0, 1)]

    # Get agents
    assert len(grid.get_agents()) == 5
    assert len(grid.get_agents([0, 0])) == 2  # Single pos
    assert len(grid.get_agents([(0, 0), (0, 1)])) == 3  # Area

    # Get items
    assert [x for x in grid.items()][0][0] == (0, 0)
    assert len([x for x in grid.items()][0][1]) == 2

    # Wrong area
    with pytest.raises(ValueError):
        assert grid.get_agents(1)


def test_attribute():
    model = ap.Model()
    model.add_grid((2, 2))
    model.env.add_agents(5)
    assert model.env.attribute('id') == [[8, 3], [4, 5]]
    assert model.env.attribute('id', sum_values=False) == [[[2, 6], [3]],
                                                           [[4], [5]]]


def test_apply():

    model = ap.Model()
    model.add_grid((3, 3))
    model.env.add_agents(1, positions=[(1, 1)])
    # Apply function len to each position, must be 1 where agent is
    assert model.env.apply(len) == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_movement():

    model = ap.Model()
    grid = model.add_grid((2, 2))
    grid.add_agents(2)
    agent = model.get_obj(2)

    assert grid.attribute('id') == [[2, 3], [np.nan, np.nan]]
    agent.move_to((1, 0))  # Move in absolute terms
    assert grid.attribute('id') == [[np.nan, 3], [2, np.nan]]
    agent.move_by((-1, 0), env=grid.id)  # Move in relative terms
    assert grid.attribute('id') == [[2, 3], [np.nan, np.nan]]
    grid.move_agent(2, (1, 0))  # Move through grid call by id
    assert grid.attribute('id') == [[np.nan, 3], [2, np.nan]]

    assert agent.neighbors() == [model.get_obj(3)]


def test_neighbors():

    model = ap.Model()
    grid = model.add_grid((5, 5))
    grid.add_agents(25)
    grid.attribute('id')
    a = model.agents[12]

    assert a.neighbors() == grid.neighbors(a.id)
    assert len(a.neighbors(distance=1)) == 8
    assert len(a.neighbors(distance=2)) == 24
    assert len(a.neighbors(diagonal=False)) == 4
