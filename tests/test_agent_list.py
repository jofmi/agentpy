import pytest
import agentpy as ap

from agentpy.tools import AgentpyError


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
    """ Select subsets with boolean operators """

    model = ap.Model()
    model.add_agents(3)

    selection1 = model.agents.id == 1
    selection2 = model.agents.id != 1
    selection3 = model.agents.id < 1
    selection4 = model.agents.id > 1
    selection5 = model.agents.id <= 1
    selection6 = model.agents.id >= 1

    assert selection1 == [False, True, False]
    assert selection2 == [True, False, True]
    assert selection3 == [True, False, False]
    assert selection4 == [False, False, True]
    assert selection5 == [True, True, False]
    assert selection6 == [False, True, True]

    assert model.agents(selection1) == model.agents.select(selection1)
    assert list(model.agents(selection1).id) == [1]


def test_sort():

    model = ap.Model()
    model.add_agents(2)
    model.agents[0].x = 1
    model.agents[1].x = 0
    model.agents.sort('x')

    assert list(model.agents.x) == [0, 1]
    assert list(model.agents.id) == [1, 0]


def test_arithmetics():

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

    agents.x *= 2
    assert list(agents.x) == [4, 6, 8]

