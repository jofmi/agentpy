import pytest
import agentpy as ap
import numpy as np
import pandas as pd
import shutil
import os

from agentpy.tools import AgentpyError


repr = """DataDict {
'parameters': 
    'fixed': Dictionary with 1 key
    'varied': DataFrame with 1 variable and 10 rows
'log': Dictionary with 5 keys
'measures': DataFrame with 1 variable and 10 rows
'variables': 
    'Agent': DataFrame with 1 variable and 10 rows
    'MyModel': DataFrame with 1 variable and 10 rows
}"""


class MyModel(ap.Model):
    def step(self):
        self.measure('x', self.p.x)
        self.add_agents()
        self.agents.record('id')
        self.record('id')
        self.stop()


def test_repr():
    param_ranges = {'x': (0., 1.), 'y': 1}
    sample = ap.sample(param_ranges, n=10)
    results = ap.Experiment(MyModel, sample, record=True).run()
    assert results.__repr__() == repr


class AgentType1(ap.Agent):
    def setup(self):
        self.x = 'x1'

    def action(self):
        self.record('x')


class AgentType2(AgentType1):
    def setup(self):
        self.x = 'x2'
        self.y = 'y2'

    def action(self):
        self.record(['x', 'y'])


class EnvType3(ap.Environment):
    def setup(self):
        self.x = 'x3'
        self.z = 'z4'

    def action(self):
        self.record(['x', 'z'])


class EnvType4(ap.Environment):
    def setup(self):
        self.z = 'z4'

    def action(self):
        self.record(['z'])


class ModelType0(ap.Model):

    def setup(self):
        self.E31 = self.add_env(env_class=EnvType3)
        self.E41 = self.add_env(env_class=EnvType4)
        self.E42 = self.add_env(env_class=EnvType4)
        self.envs.add_agents(agents=2, agent_class=AgentType1)
        self.E42.add_agents(agents=2, agent_class=AgentType2)

    def step(self):
        self.agents.action()
        self.envs.action()

    def end(self):
        self.measure('m_key', 'm_value')


def test_testing_model():

    parameters = {'steps': 2, 'px': (1, 2)}
    sample = ap.sample_discrete(parameters)
    settings = {'iterations': 2,
                'scenarios': ['test1', 'test2'],
                'record': True}

    pytest.model_instance = model = ModelType0(sample[0])
    pytest.model_results = model_results = model.run(display=False)

    exp = ap.Experiment(ModelType0, sample, **settings)
    pytest.exp_results = exp_results = exp.run(display=False)

    type_list = ['AgentType1', 'AgentType2', 'EnvType3', 'EnvType4']
    assert list(model_results.variables.keys()) == type_list
    assert list(exp_results.variables.keys()) == type_list


def arrange_things(results):

    return (results.arrange(variables='x'),
            results.arrange(variables=['x']),
            results.arrange(variables=['x', 'y']),
            results.arrange(variables='z'),
            results.arrange(parameters='px'),
            results.arrange(measures='m_key'),
            results.arrange(variables='all',
                            parameters='all',
                            measures='all'))


def test_datadict_arrange_for_single_run():

    results = pytest.model_results
    data = arrange_things(results)
    x_data, x_data2, xy_data, z_data, p_data, m_data, all_data = data

    assert x_data.equals(x_data2)
    assert list(x_data['x']) == ['x1'] * 4 + ['x2'] * 4 + ['x3'] * 2

    assert x_data.shape == (10, 4)
    assert xy_data.shape == (10, 5)
    assert z_data.shape == (6, 4)
    assert p_data.shape == (1, 2)
    assert m_data.shape == (1, 2)
    assert all_data.shape == (15, 10)


def test_datadict_arrange_for_multi_run():

    results = pytest.exp_results
    data = arrange_things(results)
    x_data, x_data2, xy_data, z_data, p_data, m_data, all_data = data

    assert x_data.equals(x_data2)
    assert x_data.shape == (80, 6)
    assert xy_data.shape == (80, 7)
    assert z_data.shape == (48, 6)
    assert p_data.shape == (4, 2)
    assert m_data.shape == (8, 3)
    assert all_data.shape == (120, 11)


def test_datadict_arrange_measures():

    results = pytest.exp_results
    mvp_data = results.arrange(measures='all', parameters='varied')
    mvp_data_2 = results.arrange_measures()
    assert mvp_data.equals(mvp_data_2)


def test_datadict_arrange_variables():

    results = pytest.exp_results
    mvp_data = results.arrange(variables='all', parameters='varied')
    mvp_data_2 = results.arrange_variables()
    assert mvp_data.equals(mvp_data_2)


def test_automatic_loading():

    if 'ap_output' in os.listdir():
        shutil.rmtree('ap_output')

    results = pytest.model_results
    results.log['test'] = False
    results.save(exp_name="a")
    results.save(exp_name="b", exp_id=1)
    results.log['test'] = True
    results.save(exp_name="b", exp_id=3)
    results.log['test'] = False
    results.save(exp_name="c")
    results.save(exp_name="b", exp_id=2)

    loaded = ap.load()
    shutil.rmtree('ap_output')

    # Latest experiment is chosen (b),
    # and then highest id is chosen (3)

    assert loaded.log['test'] is True


def test_saved_equals_loaded():

    results = pytest.exp_results
    results.save()
    loaded = ap.load('ModelType0')
    shutil.rmtree('ap_output')
    assert results == loaded
    # Test that equal doesn't hold if parts are changed
    assert results != 1
    loaded.measures = 1
    assert results != loaded
    results.measures = 1
    assert results == loaded
    loaded.log = 1
    assert results != loaded
    del loaded.log
    assert results != loaded


class WeirdObject:
    pass


repr2 = """DataDict {
'i1': 1 <class 'int'>
'i2': 1 <class 'numpy.int64'>
'f1': 1.0 <class 'float'>
'f2': 1.0 <class 'numpy.float64'>
's1': 'test' <class 'str'>
's2': 'testtesttesttesttesttest...' (length 24) <class 'str'>
'l1': List with 3 entries
'l2': Object of type <class 'numpy.ndarray'>
'wo': Object of type <class 'tests.test_output.WeirdObject'>
}"""


repr3 = """DataDict {
'f1': 1.0 <class 'float'>
'f2': 1.0 <class 'float'>
'i1': 1 <class 'int'>
'i2': 1 <class 'int'>
'l1': List with 3 entries
'l2': List with 3 entries
's1': 'test' <class 'str'>
's2': 'testtesttesttesttesttest...' (length 24) <class 'str'>
}"""


def test_save_load():

    dd = ap.DataDict()
    dd['i1'] = 1
    dd['i2'] = np.int64(1)
    dd['f1'] = 1.
    dd['f2'] = np.float64(1.)
    dd['s1'] = 'test'
    dd['s2'] = 'testtesttesttesttesttest'
    dd['l1'] = [1, 2, [3, 4]]
    dd['l2'] = np.array([1, 2, 3])
    dd['wo'] = WeirdObject()

    dd.save()
    dl = ap.load()
    with pytest.raises(FileNotFoundError):
        assert ap.load("Doesn't_exist")
    shutil.rmtree('ap_output')
    with pytest.raises(FileNotFoundError):
        assert ap.load("Doesn't_exist")

    # TODO assert dd.__repr__() == repr2
    # TODO assert dl.__repr__() == repr3
    assert len(dd) == 9
    assert len(dl) == 8
    assert dl.l1[2][1] == 4


