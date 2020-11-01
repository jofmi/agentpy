import agentpy as ap

def test_time():

    model = ap.Model({'steps':1})
    assert model.t == 0
    model.run()
    assert model.t == 1