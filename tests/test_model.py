import pytest
import numpy as np
import agentpy as ap
import random

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


def test_run_seed():
    """ Test random seed setting. """
    rd = random.Random(1)
    npseed = rd.getrandbits(128)
    nprd = np.random.default_rng(seed=npseed)
    n1 = rd.randint(0, 100)
    n2 = nprd.integers(100)

    model = ap.Model({'seed': 1})
    model.run(steps=0, display=False)
    assert model.random.randint(0, 100) == n1
    assert model.nprandom.integers(100) == n2
    model = ap.Model()
    model.run(seed=1, steps=0, display=False)
    assert model.random.randint(0, 100) == n1
    assert model.nprandom.integers(100) == n2


def test_stop():
    """ Test method Model.stop() """

    class Model(ap.Model):
        def step(self):
            if self.t == 2:
                self.stop()

    model = Model()
    model.run()

    assert model.t == 2


def test_setup():
    """ Test setup() for all object types """

    class MySetup:
        def setup(self, a):
            self.a = a + 1

    class MyAgentType(MySetup, ap.Agent):
        pass

    class MySpaceType(MySetup, ap.Space):
        pass

    class MyNwType(MySetup, ap.Network):
        pass

    class MyGridType(MySetup, ap.Grid):
        pass

    model = ap.Model()
    agents = ap.AgentList(model, 1, b=1)
    agents.extend(ap.AgentList(model, 1, MyAgentType, a=1))
    model.S1 = MySpaceType(model, shape=(1, 1), a=2)
    model.G1 = MyGridType(model, shape=(1, 1), a=3)
    model.N1 = MyNwType(model, a=4)

    # Standard setup implements keywords as attributes
    # Custom setup uses only keyword a and adds 1
    assert agents[0].b == 1
    assert agents[1].a == 2
    assert model.S1.a == 3
    assert model.G1.a == 4
    assert model.N1.a == 5


def test_create_output():
    """ Should put variables directly into output if there are only model
    variables, or make a subdict if there are also other variables. """

    model = ap.Model()
    model.record('x', 0)
    model.run(1)
    assert list(model.output.variables.Model.keys()) == ['x']

    model = ap.Model(_run_id=(1, 2))
    model.agents = ap.AgentList(model, 1)
    model.agents.record('x', 0)
    model.record('x', 0)
    model.run(1)
    assert list(model.output.variables.keys()) == ['Agent', 'Model']

    # Run id and scenario should be added to output
    assert model.output.variables.Model.reset_index()['sample_id'][0] == 1
    assert model.output.variables.Model.reset_index()['iteration'][0] == 2
