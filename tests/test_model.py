import pytest
import agentpy as ap

from agentpy.tools import AgentpyError


def test_run():
    """ Test time limits """

    # Parameter step limit
    model = ap.Model()
    assert model.t == 0
    model.p.steps = 0
    model.run()
    assert model.t == 0
    model.p.steps = 1
    model.run()
    assert model.t == 1

    # Maximum time limit
    del model.p.steps
    model.t = 999_999
    with pytest.raises(AgentpyError):
        assert model.run()
    assert model.t == 1_000_000

    # Custom time limit
    model = ap.Model({'steps': 1})
    model.t_max = 0
    with pytest.raises(AgentpyError):
        assert model.run()
    assert model.t == 0


def test_stop():
    """ Test method Model.stop() """

    class Model(ap.Model):
        def step(self):
            if self.t == 2:
                self.stop()

    model = Model()
    model.run()

    assert model.t == 2


def test_add_agents():
    """ Add new agents to model """

    model = ap.Model()
    model.add_agents(3)

    assert len(model.agents) == 3
    assert list(model.agents.id) == [0, 1, 2]
    assert all([a.envs == ap.EnvDict() for a in model.agents])


def test_prop_objects():

    model = ap.Model()
    model.add_agents(3)
    model.add_env('E')

    assert len(model.objects) == 4
    assert model.agents[0] in model.objects
    assert model.envs['E'] in model.objects


def test_setup():
    """ Test setup() for all ABM object types """

    class MySetup:
        def setup(self, a):
            self.a = a + 1

    class MyAgentType(MySetup, ap.Agent):
        pass

    class MyEnvType(MySetup, ap.Environment):
        pass

    class MyNwType(MySetup, ap.Network):
        pass

    class MyGridType(MySetup, ap.Grid):
        pass

    model = ap.Model()
    model.add_agents(1, b=1)
    model.add_agents(1, MyAgentType, a=1)
    model.add_env('E1', MyEnvType, a=2)
    model.add_env('G1', MyGridType, shape=(1, 1), a=3)
    model.add_env('N1', MyNwType, a=4)

    # Standard setup implements keywords as attributes
    # Custom setup uses only keyword a and adds 1

    with pytest.raises(TypeError):
        assert model.add_agents(1, MyAgentType, b=1)

    assert model.agents[0].b == 1
    assert model.agents[1].a == 2
    assert model.E1.a == 3
    assert model.G1.a == 4
    assert model.N1.a == 5


def test_agent_destructor():
    """ Remove agent from model """

    model = ap.Model()
    model.add_agents(3)
    del model.agents[1]

    assert len(model.agents) == 2
    assert list(model.agents.id) == [0, 2]


def test_record():
    """ Record a dynamic variable """

    model = ap.Model()
    model.add_agents(3)
    model.var1 = 1
    model.var2 = 2
    model.record(['var1', 'var2'])

    assert len(list(model._log.keys())) == 3
    assert model._log['var1'] == [1]
    assert model._log['var2'] == [2]


def test_record_all():
    """ Record all dynamic variables automatically """

    model = ap.Model()
    model.var1 = 1
    model.var2 = 2
    model.record(model.var_keys)

    assert len(list(model._log.keys())) == 3
    assert model._log['var1'] == [1]
    assert model._log['var2'] == [2]
