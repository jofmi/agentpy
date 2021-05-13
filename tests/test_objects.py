import pytest
import agentpy as ap
from agentpy.tools import AgentpyError

def test_basics():
    model = ap.Model()
    agent = ap.Agent(model)
    agent.x = 1
    agent['y'] = 2
    assert agent['x'] == 1
    assert agent.y == 2
    assert agent.__repr__() == "Agent (Obj 1)"
    assert model.type == 'Model'
    assert model.__repr__() == "Model"
    assert isinstance(model.info, ap.tools.InfoStr)
    with pytest.raises(AttributeError):
        assert agent.z


def test_single_env_agents():

    model = ap.Model()
    env = ap.Grid(model, (10, 10))
    agent = ap.Agent(model)
    env.add_agents([agent])

    with pytest.raises(AgentpyError):
        env.add_agents([agent])

    assert env == agent.env
    assert env.positions[agent] == agent.pos

    env.remove_agents([agent])

    assert agent.env is None
    assert agent.pos is None

    with pytest.raises(AgentpyError):
        env.remove_agents([agent])


def test_multi_env_agents():

    model = ap.Model()
    env1 = ap.Grid(model, (1, 1))
    env2 = ap.Grid(model, (1, 1))
    agent = ap.MultiAgent(model)
    env1.add_agents([agent])
    env2.add_agents([agent])

    assert agent.env is agent.pos
    assert agent.env == {env1: [0, 0], env2: [0, 0]}

    env1.remove_agents([agent])

    assert agent.env == {env2: [0, 0]}

    env2.remove_agents([agent])

    assert agent.env == {}

    with pytest.raises(KeyError):
        env2.remove_agents([agent])


def test_record():
    """ Record a dynamic variable """

    model = ap.Model()
    model.var1 = 1
    model.var2 = 2
    model.record(['var1', 'var2'])
    model.record('var3', 3)

    assert len(list(model.log.keys())) == 3 + 1  # One for time
    assert model.log['var1'] == [1]
    assert model.log['var2'] == [2]
    assert model.log['var3'] == [3]


def test_record_all():
    """ Record all dynamic variables """

    model = ap.Model()
    model.var1 = 1
    model.var2 = 2
    model.record(model.vars)

    assert len(list(model.log.keys())) == 3
    assert model.log['var1'] == [1]
    assert model.log['var2'] == [2]

