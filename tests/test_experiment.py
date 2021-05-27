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

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    pool = mp.Pool(mp.cpu_count())
    results = exp.run(pool)

    exp2 = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    results2 = exp2.run()

    del results.info
    del results2.info

    assert results == results2


def test_interactive():
    """Test only for errors."""
    def interactive_plot(m):
        print("x =", m.p.x)
    param_ranges = {'steps': 1, 'x': ap.Range(0., 1.)}
    sample = ap.Sample(param_ranges, n=10)
    exp = ap.Experiment(ap.Model, sample)
    exp.interactive(interactive_plot)
    assert True


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

    parameters = {
        'steps': 0,
        'seed': ap.Values(1, 1, 2)
    }

    class Model(ap.Model):
        def setup(self):
            self.report('x', self.model.random.random())

    sample = ap.Sample(parameters)
    exp = ap.Experiment(Model, sample, iterations=2, randomize=False)
    results = exp.run()

    l = list(results.reporters['x'])

    assert l[0] == l[1]
    assert l[0:2] == l[2:4]
    assert l[0:2] != l[4:6]