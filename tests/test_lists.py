import pytest
import agentpy as ap
import numpy as np


def test_repr():
    model = ap.Model()
    model.add_agents()
    model.add_env()
    assert model.agents.__repr__() == "AgentList [1 agent]"
    assert model.envs.__repr__() == "EnvList [1 environment]"
    assert model.objects.__repr__() == "ObjList [2 objects]"
    l1 = model.agents.id
    l2 = l1 + 1
    assert l1.__repr__() == "AttrList of attribute 'id': [1]"
    assert l2.__repr__() == "AttrList: [2]"


def test_attr_calls():
    model = ap.Model()
    model.add_agents(2)
    model.agents.x = 1
    model.agents.f = lambda: 2
    assert list(model.agents.x) == [1, 1]
    assert list(model.agents.f()) == [2, 2]
    with pytest.raises(AttributeError):
        assert model.agents.y
    with pytest.raises(TypeError):
        assert model.agents.x()  # noqa


def test_select():
    """ Select subsets with boolean operators. """
    model = ap.Model()
    model.add_agents(3)
    selection1 = model.agents.id == 2
    selection2 = model.agents.id != 2
    selection3 = model.agents.id < 2
    selection4 = model.agents.id > 2
    selection5 = model.agents.id <= 2
    selection6 = model.agents.id >= 2
    assert selection1 == [False, True, False]
    assert selection2 == [True, False, True]
    assert selection3 == [True, False, False]
    assert selection4 == [False, False, True]
    assert selection5 == [True, True, False]
    assert selection6 == [False, True, True]
    assert model.agents(selection1) == model.agents.select(selection1)
    assert list(model.agents(selection1).id) == [2]


def test_random():
    """ Test random shuffle and selection. """
    model = ap.Model()
    model.add_agents(2)
    assert len(model.agents) == len(model.agents.shuffle())
    assert len(model.agents.random()) == 1

    # Custom generator with seperate seed
    model = ap.Model()
    model.add_agents(5)
    generator = np.random.default_rng(1)
    assert len(model.agents.random(generator=generator)) == 1
    assert model.agents.random(generator=generator).id[0] == 3
    assert list(model.agents.shuffle(generator=generator).id) == [5, 1, 3, 2, 4]

    # Test with single agent
    model = ap.Model()
    agents = model.add_agents(1)
    assert model.agents.random()[0] is agents[0]
    assert model.agents.shuffle()[0] is agents [0]

    # Agentlist with no model defined directly
    model = ap.Model()
    agents = model.add_agents(3)
    agents = ap.AgentList(agents)
    model.run(steps=0, seed=1, display=False)
    assert agents.random()[0].id == 2

    # Agentlist with no model defined
    # (no seed control without model, test can only check if no errors)
    agents1 = ap.AgentList([1, 2, 3])
    agents1.random()


def test_sort():
    """ Test sorting method. """
    model = ap.Model()
    model.add_agents(2)
    model.agents[0].x = 1
    model.agents[1].x = 0
    model.agents.sort('x')
    assert list(model.agents.x) == [0, 1]
    assert list(model.agents.id) == [2, 1]


def test_arithmetics():
    """ Test arithmetic operators """

    model = ap.Model()
    model.add_agents(3)
    agents = model.agents

    agents.x = 1
    assert agents.x.attr == "x"
    assert list(agents.x) == [1, 1, 1]

    agents.y = ap.AttrList([1, 2, 3])
    assert list(agents.y) == [1, 2, 3]

    agents.x = agents.x + agents.y
    assert list(agents.x) == [2, 3, 4]

    agents.x = agents.x - ap.AttrList([1, 1, 1])
    assert list(agents.x) == [1, 2, 3]

    agents.x += 1
    assert list(agents.x) == [2, 3, 4]

    agents.x -= 1
    assert list(agents.x) == [1, 2, 3]

    agents.x *= 2
    assert list(agents.x) == [2, 4, 6]

    agents.x = agents.x * agents.x
    assert list(agents.x) == [4, 16, 36]

    agents.x = agents.x / agents.x
    assert list(agents.x)[0] == pytest.approx(1.)

    agents.x /= 2
    assert list(agents.x)[0] == pytest.approx(0.5)
