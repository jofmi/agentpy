import pytest
import agentpy as ap
import numpy as np
from agentpy.tools import AgentpyError


def test_basics():
    model = ap.Model()
    l1 = ap.AgentList(model, 0)
    l2 = ap.AgentList(model, 1)
    l3 = ap.AgentList(model, 2)
    assert l1.__repr__() == "AgentList (0 objects)"
    assert l2.__repr__() == "AgentList (1 object)"
    assert l3.__repr__() == "AgentList (2 objects)"

    agentlist = ap.AgentList(model, 2)
    AgentDList = ap.AgentDList(model, 2)
    agentiter = ap.AgentIter(model, agentlist)
    AgentDListiter = ap.AgentDListIter(agentlist)
    attriter = agentiter.id

    assert np.array(agentlist).tolist() == list(agentlist)
    assert np.array(AgentDList).tolist() == list(AgentDList)
    assert np.array(agentiter).tolist() == list(agentiter)
    assert np.array(AgentDListiter).tolist() == list(AgentDListiter)
    assert np.array(attriter).tolist() == list(attriter)

    with pytest.raises(AgentpyError):
        _ = agentiter[2]  # No item lookup allowed

    # Seta and get attribute
    for agents in [agentlist, AgentDList, agentiter]:
        agents.x = 1
        agents.y = 1
        assert list(agents.x) == [1, 1]
        agents.x += agents.x
        assert list(agents.x) == [2, 2]


def test_kwargs():
    model = ap.Model()
    attrs = [1, 2]
    agents = ap.AgentList(model, 2, x=attrs)

    assert list(agents.x) == [[1, 2], [1, 2]]

    model = ap.Model()
    attrs = ap.AttrIter([1, 2])
    agents = ap.AgentList(model, 2, x=attrs)

    assert list(agents.x) == [1, 2]

    with pytest.raises(AgentpyError):
        agents = ap.AgentList(model, 2, ap.Agent, 'arg w/o keyword')


def test_add():
    model = ap.Model()
    agents1 = ap.AgentList(model, 2)
    agents2 = ap.AgentList(model, 2)
    agents3 = agents1 + agents2

    assert list(agents3.id) == [1, 2, 3, 4]

    agents4 = agents3 + [ap.Agent(model)]

    assert list(agents4.id) == [1, 2, 3, 4, 5]

    model = ap.Model()
    agents1 = ap.AgentDList(model, 2)
    agents2 = ap.AgentDList(model, 2)
    agents3 = agents1 + agents2

    assert list(agents3.id) == [1, 2, 3, 4]

    agents4 = agents3 + [ap.Agent(model)]

    assert list(agents4.id) == [1, 2, 3, 4, 5]


def test_agent_group():
    class MyAgent(ap.Agent):
        def method(self, x):
            if self.id == 2:
                self.model.agents.pop(x)
            self.model.called.append(self.id)

    # Delete later element in list
    model = ap.Model()
    model.called = []
    model.agents = ap.AgentDList(model, 4, MyAgent)
    model.agents.buffer().method(2)
    assert model.called == [1, 2, 4]

    # Delete earlier element in list
    model = ap.Model()
    model.called = []
    model.agents = ap.AgentDList(model, 4, MyAgent)
    model.agents.buffer().method(0)
    assert model.called == [1, 2, 3, 4]

    # Incorrect result without buffer
    model = ap.Model()
    model.called = []
    model.agents = ap.AgentList(model, 4, MyAgent)
    model.agents.method(0)
    assert model.called == [1, 2, 4]

    # Combine with buffer - still number 3 that gets deleted
    model = ap.Model()
    model.run(seed=2, steps=0, display=False)
    model.called = []
    model.agents = ap.AgentDList(model, 4, MyAgent)
    model.agents.shuffle().buffer().method(2)
    assert model.called == [2, 4, 1]

    # Combination order doesn't matter
    model = ap.Model()
    model.run(seed=2, steps=0, display=False)
    model.called = []
    model.agents = ap.AgentDList(model, 4, MyAgent)
    model.agents.buffer().shuffle().method(2)
    assert model.called == [2, 4, 1]


def test_attr_list():
    model = ap.Model()
    model.agents = ap.AgentList(model, 2)
    model.agents.id[1] = 5
    assert model.agents.id[1] == 5
    model.agents.x = 1
    model.agents.f = lambda: 2
    assert list(model.agents.x) == [1, 1]
    assert list(model.agents.f()) == [2, 2]
    with pytest.raises(AttributeError):
        assert list(model.agents.y)  # Convert to list to call attribute
    with pytest.raises(TypeError):
        assert model.agents.x()  # noqa

    model = ap.Model()
    l3 = ap.AgentList(model, 2)
    assert l3.id == [1, 2]
    assert l3.id.__repr__() == "[1, 2]"
    assert l3.p.update({1: 1}) == [None, None]
    assert l3.p == [{1: 1}, {1: 1}]

    # Attribute list with attribute key
    # sets/gets attr like a normal list
    al = l3.id + 1
    al[1] = 0
    assert al[1] == 0


def test_select():
    """ Select subsets with boolean operators. """
    model = ap.Model()
    model.agents = ap.AgentList(model, 3)
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
    assert list(model.agents.select(selection1).id) == [2]

    model = ap.Model()
    model.agents = ap.AgentDList(model, 3)
    selection1 = model.agents.id == 2
    assert selection1 == [False, True, False]
    assert list(model.agents.select(selection1).id) == [2]


def test_random():
    """ Test random shuffle and selection. """
    # Agent List
    model = ap.Model()
    model.run(steps=0, seed=1, display=False)
    model.agents = ap.AgentList(model, 10)
    assert list(model.agents.random())[0].id == 2
    assert model.agents.shuffle()[0].id == 9
    assert list(model.agents.random(11, replace=True).id)[0] == 6
    assert list(model.agents.random(2).id) == [9, 5]

    # Test with single agent
    model = ap.Model()
    agents = ap.AgentList(model, 1)
    assert agents.shuffle()[0] is agents[0]
    assert list(agents.random())[0] is agents[0]

    # Agent Group
    model = ap.Model()
    model.run(steps=0, seed=1, display=False)
    model.agents = ap.AgentDList(model, 10)
    assert list(model.agents.random())[0].id == 2
    assert list(model.agents.shuffle())[0].id == 9


def test_sort():
    """ Test sorting method. """
    model = ap.Model()
    model.agents = ap.AgentList(model, 2)
    model.agents[0].x = 1
    model.agents[1].x = 0
    model.agents.sort('x')
    assert list(model.agents.x) == [0, 1]
    assert list(model.agents.id) == [2, 1]

    model = ap.Model()
    model.agents = ap.AgentDList(model, 2)
    model.agents[0].x = 1
    model.agents[1].x = 0
    model.agents = model.agents.sort('x')  # Not in-place
    assert list(model.agents.x) == [0, 1]
    assert list(model.agents.id) == [2, 1]


def test_arithmetics():
    """ Test arithmetic operators """

    model = ap.Model()
    model.agents = ap.AgentList(model, 3)
    agents = model.agents

    agents.x = 1
    assert agents.x.attr == "x"
    assert list(agents.x) == [1, 1, 1]

    agents.y = ap.AttrIter([1, 2, 3])
    assert list(agents.y) == [1, 2, 3]

    agents.x = agents.x + agents.y
    assert list(agents.x) == [2, 3, 4]

    agents.x = agents.x - ap.AttrIter([1, 1, 1])
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


def test_remove():
    model = ap.Model()
    agents = ap.AgentList(model, 3, ap.Agent)
    assert list(agents.id) == [1, 2, 3]
    agents.remove(agents[0])
    assert list(agents.id) == [2, 3]

    model = ap.Model()
    agents = ap.AgentDList(model, 3, ap.Agent)
    assert list(agents.id) == [1, 2, 3]
    agents.remove(agents[0])
    assert list(agents.id) == [3, 2]

    model = ap.Model()
    agents = ap.AgentDList(model, 3, ap.Agent)
    assert list(agents.id) == [1, 2, 3]
    agents.pop(0)
    assert list(agents.id) == [3, 2]

    model = ap.Model()
    agents = ap.AgentSet(model, 3, ap.Agent)
    assert set(agents.id) == set([1, 2, 3])
    agents.remove(next(iter(agents)))
    assert len(agents.id) == 2