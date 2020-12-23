import pytest
import agentpy as ap
from agentpy.tools import AgentpyError


def test_apply():

    model = ap.Model()
    model.add_grid((3, 3))
    model.env.add_agents(1, positions=[(1, 1)])

    assert model.env.apply(len) == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
