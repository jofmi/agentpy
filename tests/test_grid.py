import pytest
import agentpy as ap
import numpy as np


def test_apply():

    model = ap.Model()
    model.add_grid((3, 3))
    model.env.add_agents(1, positions=[(1, 1)])

    assert model.env.apply(len) == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_movement():

    model = ap.Model()
    grid = model.add_grid((2, 2))
    grid.add_agents(2)
    agent = model.get_obj(2)

    assert grid.attribute('id') == [[2, 3], [np.nan, np.nan]]
    agent.move_to((1, 0))
    assert grid.attribute('id') == [[np.nan, 3], [2, np.nan]]
    agent.move_by((-1, 0))
    assert grid.attribute('id') == [[2, 3], [np.nan, np.nan]]

    assert agent.neighbors() == [model.get_obj(3)]
