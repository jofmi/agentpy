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


def test_agent_errors():

    model = ap.Model()
    env = model.add_env()
    agent = model.add_agents()[0]
    with pytest.raises(AgentpyError):
        agent.neighbors()
    with pytest.raises(AgentpyError):
        agent.neighbors(env)

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


def test_enter_exit():

    model = ap.Model()
    model.add_agents()
    agent = model.get_obj(1)
    env = model.add_env()

    assert len(env.agents) == 0
    agent.enter(env)
    assert len(env.agents) == 1
    agent.exit(env)
    assert len(env.agents) == 0
    agent.enter(2)
    assert len(env.agents) == 1
    agent.exit(2)
    assert len(env.agents) == 0
    agent.enter(2)
    assert len(env.agents) == 1
    agent.exit()  # Take only env by default
    assert len(env.agents) == 0
    with pytest.raises(AgentpyError):
        agent.exit()  # Agent is not part of any environment
    with pytest.raises(AgentpyError):
        agent.exit(2)  # Agent is not part of this environment


def test_remove_agents():
    """ Remove agents from environment """

    model = make_forest()
    model.env.remove_agents(model.agents)

    assert len(model.env.agents) == 0
    assert len(model.agents) == 3

    model.remove_agents(model.agents)

    assert len(model.agents) == 0
