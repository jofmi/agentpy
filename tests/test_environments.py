import pytest
import agentpy as ap

from agentpy.tools import AgentpyError


def make_forest():

    model = ap.Model()
    model.add_env('forest', color='green')
    model.forest.add_agents(3)

    return model


def test_add_env():
    """ Add environment to model """

    model = make_forest()

    assert len(model.envs) == 1
    assert model.forest.key == 'forest'
    assert model.forest.color == 'green'
    assert type(model.forest) == ap.Environment
    assert model.forest == model.envs['forest']
    assert model.agents == model.forest.agents
    assert model.agents[0].envs == model.envs


def test_exit_env():
    """ Remove agent from environment """

    model = make_forest()
    model.agents[-1].exit('forest')

    assert len(model.forest.agents) == 2
    assert len(model.agents) == 3
    assert list(model.forest.agents.id) == [0, 1]
