import pytest
import agentpy as ap
from agentpy.tools import AgentpyError


def make_model():
    """ Create three agents (ID 1-3) in an environment (ID 4). """
    model = ap.Model()
    agents = model.add_agents(3)
    env = model.add_env(color='green')
    env.add_agents(agents)
    return model


def test_add_env():
    """ Add environment to model. """
    model = make_model()
    assert len(model.envs) == 1
    assert model.env.id == 4  # Environment is added after agents (ID 1-3)
    assert model.env.color == 'green'  # Attribute passed as kwarg
    assert type(model.env) == ap.Environment
    assert model.env == model.envs[0]
    assert model.agents == model.env.agents
    assert model.agents[0].envs == model.envs


def test_remove_agents():
    """ Remove/delete agents from environment/model. """
    model = make_model()  # Env with three agents
    assert len(model.env.agents) == 3
    model.env.remove_agents(model.agents)  # Remove from env
    assert len(model.env.agents) == 0
    assert len(model.agents) == 3
    model.remove_agents(model.agents)  # Remove from model
    assert len(model.agents) == 0
    model = make_model()
    model.agents.delete()  # Remove from everything
    assert len(model.env.agents) == 0
    assert len(model.agents) == 0


def test_enter_exit():
    """ Move agent in and out of environment. """
    model = ap.Model()
    model.add_agents()
    agent = model.get_obj(1)
    env = model.add_env()
    assert len(env.agents) == 0
    agent.enter(env)  # Call env by instance
    assert len(env.agents) == 1
    agent.exit(env)
    assert len(env.agents) == 0
    agent.enter(2)  # Call env by id
    assert len(env.agents) == 1
    agent.exit(2)
    assert len(env.agents) == 0
    agent.enter(2)
    assert len(env.agents) == 1
    agent.exit()  # Take first env by default
    assert len(env.agents) == 0
    with pytest.raises(AgentpyError):
        agent.exit()  # Agent is not part of any environment
    with pytest.raises(AgentpyError):
        agent.exit(2)  # Agent is not part of this environment


def test_neighbors_error():
    """ Neighbors cannot be called in default environment. """
    model = ap.Model()
    env = model.add_env()
    agent = model.add_agents()[0]
    with pytest.raises(AgentpyError):
        agent.neighbors()  # Agent has no suitable environment
    with pytest.raises(AgentpyError):
        agent.neighbors(env)  # Chosen environment is not suitable
