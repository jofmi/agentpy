import pytest
import agentpy as ap

from agentpy.tools import AgentpyError


def test_time():
    """ Test time limits """

    # Passed step limit
    model = ap.Model({'steps': 1})
    assert model.t == 0
    model.run()
    assert model.t == 1

    # Maximum time limit
    del model.p.steps
    model.t = 999_999
    with pytest.raises(AgentpyError):
        assert model.run()
    assert model.t == 1_000_000


def test_add_agents():
    """ Add new agents to model """

    model = ap.Model()
    model.add_agents(3)

    assert len(model.agents) == 3
    assert list(model.agents.id) == [0, 1, 2]
    assert all([a.envs == ap.EnvDict(model) for a in model.agents])


def test_exit():
    """ Remove agent from model """

    model = ap.Model()
    model.add_agents(4)
    model.agents[3].exit()

    assert len(model.agents) == 3
    assert list(model.agents.id) == [0, 1, 2]


def test_record():
    """ Record a dynamic variable """

    model = ap.Model()
    model.var1 = 1
    model.var2 = 2
    model.record(['var1', 'var2'])

    assert model._log['var2'] == [2]

