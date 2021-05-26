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

