import pytest
import numpy as np
import agentpy as ap

from agentpy.tools import AgentpyError


def test_run():
    """ Test time limits. """

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
    model.t = 999
    model.run()
    assert model.t == 1000


def test_run_seed():
    """ Test random seed setting. """
    n = np.random.default_rng(1).integers(10)
    model = ap.Model({'seed': 1})
    model.run(steps=0, display=False)
    assert model.random.integers(10) == n
    model = ap.Model()
    model.run(seed=1, steps=0, display=False)
    assert model.random.integers(10) == n


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

    assert len(model.agents) == 3  # New agents
    assert list(model.agents.id) == [1, 2, 3]

    model.add_agents(model.agents)  # Existing agents
    assert list(model.agents.id) == [1, 2, 3] * 2


def test_objects_property():

    model = ap.Model()
    model.add_agents(3)
    model.add_env()

    assert len(model.objects) == 4
    assert model.agents[0] in model.objects
    assert model.envs[0] in model.objects


def test_setup():
    """ Test setup() for all object types """

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
    model.E1 = model.add_env(MyEnvType, a=2)
    model.G1 = model.add_env(MyGridType, shape=(1, 1), a=3)
    model.N1 = model.add_env(MyNwType, a=4)

    # Standard setup implements keywords as attributes
    # Custom setup uses only keyword a and adds 1

    with pytest.raises(TypeError):
        assert model.add_agents(1, MyAgentType, b=1)

    assert model.agents[0].b == 1
    assert model.agents[1].a == 2
    assert model.E1.a == 3
    assert model.G1.a == 4
    assert model.N1.a == 5


def test_delete():
    """ Remove agent from model """

    model = ap.Model()
    model.add_agents(3)
    model.add_env().add_agents(model.agents)
    model.agents[1].delete()

    assert len(model.agents) == 2
    assert list(model.agents.id) == [1, 3]
    assert list(model.env.agents.id) == [1, 3]


def test_create_output():
    """ Should put variables directly into output if there are only model
    variables, or make a subdict if there are also other variables. """

    model = ap.Model()
    model.record('x', 0)
    model.run(1)
    assert list(model.output.variables.keys()) == ['x']

    model = ap.Model(run_id=1, scenario='test')
    model.add_agents()
    model.agents.record('x', 0)
    model.record('x', 0)
    model.run(1)
    assert list(model.output.variables.keys()) == ['Agent', 'Model']

    # Run id and scenario should be added to output
    assert model.output.variables.Model.reset_index()['run_id'][0] == 1
    assert model.output.variables.Model.reset_index()['scenario'][0] == 'test'
