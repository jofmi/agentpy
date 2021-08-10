import pytest
import agentpy as ap
import numpy as np
import pandas as pd
import shutil
import os

from agentpy.tools import AgentpyError

from SALib.sample import saltelli
from SALib.analyze import sobol


def test_combine_vars():

    model = ap.Model()
    model.record('test', 1)
    results = model.run(1, display=False)
    assert results._combine_vars().shape == (1, 1)

    model = ap.Model()
    agents = ap.AgentList(model, 1)
    agents.record('test', 1)
    results = model.run(1, display=False)
    assert results._combine_vars().shape == (1, 1)

    model = ap.Model()
    agents = ap.AgentList(model, 1)
    model.record('test', 1)
    agents.record('test', 2)
    results = model.run(1, display=False)
    assert results._combine_vars().shape == (2, 1)

    model = ap.Model()
    agents = ap.AgentList(model, 1)
    model.record('test', 1)
    agents.record('test', 2)
    results = model.run(1, display=False)
    assert results._combine_vars(obj_types="Model").shape == (1, 1)

    model = ap.Model()
    agents = ap.AgentList(model, 1)
    model.record('test', 1)
    agents.record('test', 2)
    results = model.run(1, display=False)
    assert results._combine_vars(obj_types="Doesn't exist") is None

    model = ap.Model()
    results = model.run(1, display=False)
    assert results._combine_vars() is None
    assert results._combine_pars() is None

    model = ap.Model({'test': 1})
    results = model.run(1, display=False)
    assert results._combine_pars(constants=False) is None

    #results.variables = 1
    #with pytest.raises(TypeError):
    #    assert results._combine_vars()
    #results.parameters = 1
    #with pytest.raises(TypeError):
    #    assert results._combine_pars()


repr = """DataDict {
'info': Dictionary with 12 keys
'parameters': 
    'constants': Dictionary with 1 key
    'sample': DataFrame with 1 variable and 10 rows
    'log': Dictionary with 3 keys
'variables': 
    'Agent': DataFrame with 1 variable and 10 rows
    'MyModel': DataFrame with 1 variable and 10 rows
'reporters': DataFrame with 1 variable and 10 rows
}"""


class MyModel(ap.Model):
    def step(self):
        self.report('x', self.p.x)
        self.agents = ap.AgentList(self, 1)
        self.agents.record('id')
        self.record('id')
        self.stop()


def test_repr():
    param_ranges = {'x': ap.Range(0., 1.), 'y': 1}
    sample = ap.Sample(param_ranges, n=10)
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


class EnvType3(ap.Agent):
    def setup(self):
        self.x = 'x3'
        self.z = 'z4'

    def action(self):
        self.record(['x', 'z'])


class EnvType4(ap.Agent):
    def setup(self):
        self.z = 'z4'

    def action(self):
        self.record(['z'])


class ModelType0(ap.Model):

    def setup(self):
        self.E31 = EnvType3(self)
        self.E41 = EnvType4(self)
        self.E42 = EnvType4(self)
        self.agents1 = ap.AgentList(self, 2, AgentType1)
        self.agents2 = ap.AgentList(self, 2, AgentType2)
        self.agents = ap.AgentList(self, self.agents1 + self.agents2)
        self.envs = ap.AgentList(self, [self.E31, self.E41, self.E42])

    def step(self):
        self.agents.action()
        self.envs.action()

    def end(self):
        self.report('m_key', 'm_value')


def test_testing_model():

    parameters = {'steps': 2, 'px': ap.Values(1, 2)}
    sample = ap.Sample(parameters)
    settings = {'iterations': 2,
                'record': True}

    pytest.model_instance = model = ModelType0(list(sample)[0])
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
            results.arrange(reporters='m_key'),
            results.arrange(variables=True,
                            parameters=True,
                            reporters=True),
            results.arrange())


def test_datadict_arrange_for_single_run():

    results = pytest.model_results
    data = arrange_things(results)
    x_data, x_data2, xy_data, z_data, p_data, m_data, all_data, no_data = data

    assert x_data.equals(x_data2)
    assert list(x_data['x']) == ['x1'] * 4 + ['x2'] * 4 + ['x3'] * 2

    assert x_data.shape == (10, 4)
    assert xy_data.shape == (10, 5)
    assert z_data.shape == (6, 4)
    assert p_data.shape == (1, 2)
    assert m_data.shape == (1, 2)
    assert all_data.shape == (15, 8)
    assert no_data.empty is True


def test_datadict_arrange_for_multi_run():

    results = pytest.exp_results
    data = arrange_things(results)
    x_data, x_data2, xy_data, z_data, p_data, m_data, all_data, no_data = data

    assert x_data.equals(x_data2)
    assert x_data.shape == (40, 6)
    assert xy_data.shape == (40, 7)
    assert z_data.shape == (24, 6)
    assert p_data.shape == (2, 2)
    assert m_data.shape == (4, 3)
    assert all_data.shape == (60, 10)
    assert no_data.empty is True


def test_datadict_arrange_measures():

    results = pytest.exp_results
    mvp_data = results.arrange(reporters=True, parameters=True)
    mvp_data_2 = results.arrange_reporters()
    assert mvp_data.equals(mvp_data_2)


def test_datadict_arrange_variables():

    results = pytest.exp_results
    mvp_data = results.arrange(variables=True, parameters=True)
    mvp_data_2 = results.arrange_variables()
    assert mvp_data.equals(mvp_data_2)


def test_automatic_loading():

    if 'ap_output' in os.listdir():
        shutil.rmtree('ap_output')

    results = pytest.model_results
    results.info['test'] = False
    results.save(exp_name="a")
    results.save(exp_name="b", exp_id=1)
    results.info['test'] = True
    results.save(exp_name="b", exp_id=3)
    results.info['test'] = False
    results.save(exp_name="c")
    results.save(exp_name="b", exp_id=2)

    loaded = ap.DataDict.load()
    shutil.rmtree('ap_output')

    # Latest experiment is chosen (b),
    # and then highest id is chosen (3)

    assert loaded.info['test'] is True


def test_saved_equals_loaded():

    results = pytest.exp_results
    results.save()
    loaded = ap.DataDict.load('ModelType0')
    shutil.rmtree('ap_output')
    assert results == loaded
    # Test that equal doesn't hold if parts are changed
    assert results != 1
    loaded.reporters = 1
    assert results != loaded
    results.reporters = 1
    assert results == loaded
    loaded.info = 1
    assert results != loaded
    del loaded.info
    assert results != loaded


class WeirdObject:
    pass


def test_save_load():

    dd = ap.DataDict()
    dd['i1'] = 1
    dd['i2'] = np.int64(1)
    dd['f1'] = 1.
    dd['f2'] = np.float32(1.1)
    dd['s1'] = 'test'
    dd['s2'] = 'testtesttesttesttesttest'
    dd['l1'] = [1, 2, [3, 4]]
    dd['l2'] = np.array([1, 2, 3])
    dd['wo'] = WeirdObject()

    dd.save()
    dl = ap.DataDict.load()
    with pytest.raises(FileNotFoundError):
        assert ap.DataDict.load("Doesn't_exist")
    shutil.rmtree('ap_output')
    with pytest.raises(FileNotFoundError):
        assert ap.DataDict.load("Doesn't_exist")

    assert dd.__repr__().count('\n') == 10
    assert dl.__repr__().count('\n') == 9
    assert len(dd) == 9
    assert len(dl) == 8
    assert dl.l1[2][1] == 4


def test_load_unreadable():
    """ Unreadable entries are loaded as None. """
    path = f'ap_output/fake_experiment_1/'
    os.makedirs(path)
    f = open(path + "unreadable_entry.xxx", "w+")
    f.close()
    dl = ap.DataDict.load()
    shutil.rmtree('ap_output')
    assert dl.unreadable_entry is None


class SobolModel(ap.Model):
    def step(self):
        self.report('x', self.p.x)
        self.stop()


def test_calc_sobol():
    # Running a demo problem with salib
    problem = {'num_vars': 1, 'names': ['x'], 'bounds': [[0, 1]]}
    param_values = saltelli.sample(problem, 8)
    si = sobol.analyze(problem, param_values.T[0])['S1']

    parameters = {'x': ap.Range(0., 1.)}
    sample = ap.Sample(parameters, n=8, method='saltelli', calc_second_order=False)
    results = ap.Experiment(SobolModel, sample).run(display=False)
    results.calc_sobol(reporters='x')
    assert results.sensitivity.sobol['S1'][0] == si

    # Test if a non-varied parameter causes errors
    parameters = {'x': ap.Range(0., 1.), 'y': 1}
    sample = ap.Sample(parameters, n=8, method='saltelli', calc_second_order=False)
    results = ap.Experiment(SobolModel, sample).run(display=False)
    results.calc_sobol()
    assert results.sensitivity.sobol['S1'][0] == si

    # Test wrong sample type raises error
    parameters = {'x': ap.Range(0., 1.), 'y': 1}
    sample = ap.Sample(parameters, n=8)
    results = ap.Experiment(SobolModel, sample).run(display=False)
    with pytest.raises(AgentpyError):
        results.calc_sobol()

    # Test merging iterations
    # TODO Improve
    parameters = {'x': ap.Range(0., 1.)}
    sample = ap.Sample(parameters, n=8, method='saltelli', calc_second_order=False)
    results = ap.Experiment(SobolModel, sample, iterations=10).run(display=False)
    results.calc_sobol(reporters='x')
    assert results.sensitivity.sobol['S1'][0] == si

    # Test calc_second_order
    parameters = {'x': ap.Range(0., 1.), 'y': 1}
    sample = ap.Sample(parameters, n=8, method='saltelli', calc_second_order=True)
    results = ap.Experiment(SobolModel, sample).run(display=False)
    results.calc_sobol()
    assert results.sensitivity.sobol[('S2', 'x')][0].__repr__() == 'nan'
