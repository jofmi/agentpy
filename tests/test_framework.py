import agentpy as ap


def test_time():
    """ Test time limits """

    # Passed step limit
    model = ap.Model({'steps': 1})
    assert model.t == 0
    model.run()
    assert model.t == 1

    # Maximum time
    del model.p.steps
    model.t = 999_999
    model.run()
    assert model.t == 1_000_000


def test_add_agents():
    """ Add new agents to model """

    model = ap.Model()
    model.add_agents(3)

    assert len(model.agents) == 3
    assert list(model.agents.id) == [0, 1, 2]
    assert all([a.envs == ap.EnvDict(model) for a in model.agents])


def test_add_env():
    """ Add environment to model """

    model = ap.Model()
    model.add_env('forest')
    model.forest.add_agents()

    assert len(model.envs) == 1
    assert model.forest.key == 'forest'
    assert type(model.forest) == ap.Environment
    assert model.forest == model.envs['forest']
    assert model.agents == model.forest.agents
    assert model.agents[0].envs == model.envs


def test_exit():
    """ Remove agent from model """

    model = ap.Model()
    model.add_agents(4)
    model.agents[3].exit()

    assert len(model.agents) == 3
    assert list(model.agents.id) == [0, 1, 2]


def test_exit_env():
    """ Remove agent from environment """

    model = ap.Model()
    model.add_env('forest')
    model.forest.add_agents(4)
    model.agents[-1].exit('forest')

    assert len(model.forest.agents) == 3
    assert len(model.agents) == 4
    assert list(model.forest.agents.id) == [0, 1, 2]
