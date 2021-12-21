import pytest
import agentpy as ap
import pandas as pd
import shutil
import os
import multiprocessing as mp

from agentpy.tools import AgentpyError


class MyModel(ap.Model):

    def setup(self):
        self.report('measured_id', self.model._run_id)
        self.record('t0', self.t)


def test_basics():

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    results = exp.run()
    assert 'variables' not in results
    assert exp.name == 'MyModel'

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3, record=True)
    results = exp.run()
    assert 'variables' in results


def test_parallel_processing():

    exp = ap.Experiment(MyModel, [{'steps': 1, 'report_seed': False}] * 3)
    pool = mp.Pool(mp.cpu_count())
    results = exp.run(pool=pool)

    exp2 = ap.Experiment(MyModel, [{'steps': 1, 'report_seed': False}] * 3)
    results2 = exp2.run()

    exp3 = ap.Experiment(MyModel, [{'steps': 1, 'report_seed': False}] * 3)
    results3 = exp3.run(n_jobs=-1)

    del results.info
    del results2.info
    del results3.info

    assert results == results2
    assert results2 == results3


def test_random():
    parameters = {
        'steps': 0,
        'seed': ap.Values(1, 1, 2)
    }

    class Model(ap.Model):
        def setup(self):
            self.report('x', self.model.random.random())

    sample = ap.Sample(parameters)
    exp = ap.Experiment(Model, sample, iterations=2, randomize=True)
    results = exp.run()

    l = list(results.reporters['x'])

    assert l[0] != l[1]
    assert l[0:2] == l[2:4]
    assert l[0:2] != l[4:6]

    sample = ap.Sample(parameters)
    exp = ap.Experiment(Model, sample, iterations=2, randomize=False)
    results = exp.run()

    l = list(results.reporters['x'])

    assert l[0] == l[1]
    assert l[0:2] == l[2:4]
    assert l[0:2] != l[4:6]

    exp = ap.Experiment(Model, parameters, iterations=2)
    results = exp.run()

    l1 = list(results.reporters['x'])

    assert l1 == [0.03542265363082542, 0.08363464439430013]

    parameters['seed'] = 1
    exp = ap.Experiment(Model, parameters, iterations=2)
    results = exp.run()

    l2 = list(results.reporters['x'])

    assert l1 == l2
