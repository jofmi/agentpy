import agentpy as ap


def test_time():

    model = ap.Model({'steps':1})
    assert model.t == 0
    model.run()
    assert model.t == 1


def test_max_time():

    model = ap.Model()
    model.t = 999_999
    model.run()
    assert model.t == 1_000_000


def test_add_agents():

    model = ap.Model()
    model.add_agents(3)
    assert len(model.agents) == 3
    assert list(model.agents.id) == [1,2,3]


def test_exit():

    model = ap.Model()
    model.add_agents(4)
    model.agents[3].exit()
    assert len(model.agents) == 3
    assert list(model.agents.id) == [1,2,3]