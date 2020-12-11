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


def test_parallel_processing():

    exp = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    pool = mp.Pool(mp.cpu_count())
    results = exp.run(pool)

    exp2 = ap.Experiment(MyModel, [{'steps': 1}] * 3)
    results2 = exp2.run()

    del results.log
    del results2.log

    assert results == results2