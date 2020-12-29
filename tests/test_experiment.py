import pytest
import agentpy as ap
import pandas as pd
import shutil
import os
import multiprocessing as mp

from agentpy.tools import AgentpyError


class MyModel(ap.Model):

    def setup(self):
        self.measure('measured_id', self.model.run_id)
        self.record('t0', self.t)


def test_basics():

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    results = exp.run()
    assert 'variables' not in results
    assert exp.name == 'MyModel'

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3, name='test')
    assert exp.name == 'test'

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3, record=True)
    results = exp.run()
    assert 'variables' in results


def test_parallel_processing():

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    pool = mp.Pool(mp.cpu_count())
    results = exp.run(pool)

    exp2 = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    results2 = exp2.run()

    del results.log
    del results2.log

    assert results == results2


def test_interactive():
    """Test only for errors."""
    def interactive_plot(m):
        print("x =", m.p.x)
    param_ranges = {'x': (0., 1.)}
    sample = ap.sample(param_ranges, n=10)
    exp = ap.Experiment(ap.Model, sample)
    exp.interactive(interactive_plot)
    assert True
