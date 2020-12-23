import pytest
import agentpy as ap

from agentpy.tools import AgentpyError


def make_forest():
    """ IDs are
    1 -> Environment
    2-4 -> Agents
    """
    model = ap.Model()
    model.add_env(color='green')
    model.env.add_agents(3)

    return model


def test_add_env():
    """ Add environment to model """

    model = make_forest()

    assert len(model.envs) == 1
    assert model.env.id == 1
    assert model.env.color == 'green'
    assert type(model.env) == ap.Environment
    assert model.env == model.envs[0]
    assert model.agents == model.env.agents
    assert model.agents[0].envs == model.envs


def test_exit_env():
    """ Remove single agent from environment """

    model = make_forest()
    model.agents[-1].exit(1)

    assert len(model.env.agents) == 2
    assert len(model.agents) == 3
    assert list(model.env.agents.id) == [2, 3]


def test_remove_agents():
    """ Remove agents from environment """

    model = make_forest()
    model.env.remove_agents(model.agents)

    assert len(model.env.agents) == 0
    assert len(model.agents) == 3

    model.remove_agents(model.agents)

    assert len(model.agents) == 0