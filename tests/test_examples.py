import pytest
import agentpy as ap
from agentpy.examples import *

# Test that examples run without errors

def test_WealthModel():
    parameters = {
        'seed': 42,
        'agents': 1000,
        'steps': 100,
    }

    model = WealthModel(parameters)
    results = model.run(display=False)
    assert model.reporters['gini'] == 0.627486

def test_SegregationModel():
    parameters = {
        'seed': 42,
        'want_similar': 0.3,
        'n_groups': 2,
        'density': 0.95,
        'size': 50
    }

    model = SegregationModel(parameters)
    results = model.run(display=False)
    assert model.reporters['segregation'] == 0.78