import pytest
import agentpy as ap

repr = """Agent-based model {
'type': Model
'agents': AgentList [1 agent]
'envs': EnvList [1 environment]
'p': AttrDict {0 entries}
't': 0
'run_id': None
'scenario': None
'output': DataDict {1 entry}
}"""


def test_basics():
    model = ap.Model()
    model.add_env()
    model.env.add_agents()
    agent = model.agents[0]
    agent.x = 1
    agent['y'] = 2
    assert agent['x'] == 1
    assert agent.y == 2
    assert model.env.__repr__() == "Environment (Obj 1)"
    assert agent.__repr__() == "Agent (Obj 2)"
    assert model.env.topology is None
    assert model.type == 'Model'
    assert model.__repr__() == repr
    with pytest.raises(AttributeError):
        assert agent.z
    assert agent is model.get_obj(2)
    with pytest.raises(ValueError):
        assert model.get_obj(3)


def test_record():
    """ Record a dynamic variable """

    model = ap.Model()
    model.add_agents(3)
    model.var1 = 1
    model.var2 = 2
    model.record(['var1', 'var2'])
    model.record('var3', 3)

    assert len(list(model._log.keys())) == 3 + 1  # One for time
    assert model._log['var1'] == [1]
    assert model._log['var2'] == [2]
    assert model._log['var3'] == [3]


def test_record_all():
    """ Record all dynamic variables automatically """

    model = ap.Model()
    model.var1 = 1
    model.var2 = 2
    model.record(model.var_keys)

    assert len(list(model._log.keys())) == 3
    assert model._log['var1'] == [1]
    assert model._log['var2'] == [2]