import pytest
import agentpy as ap

from agentpy.tools import AgentpyError


def test_add_env():
    """ Add environment to model """

    model = ap.Model()
    model.add_env('forest')
    model.forest.add_agents()

    assert len(model.envs) == 1
    assert model.forest.key == 'forest'
    assert type(model.forest) == ap.Environment
    assert model.forest == model.envs['forest']
    assert model.agents == model.forest.agents
    assert model.agents[0].envs == model.envs


def test_exit_env():
    """ Remove agent from environment """

    model = ap.Model()
    model.add_env('forest')
    model.forest.add_agents(4)
    model.agents[-1].exit('forest')

    assert len(model.forest.agents) == 3
    assert len(model.agents) == 4
    assert list(model.forest.agents.id) == [0, 1, 2]