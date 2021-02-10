import pytest
import agentpy as ap
from SALib.sample import saltelli


out = [{'a': 1, 'b': 3.0, 'c': 5},
       {'a': 1, 'b': 3.5, 'c': 5},
       {'a': 2, 'b': 3.0, 'c': 5},
       {'a': 2, 'b': 3.5, 'c': 5}]


def test_sample():
    parameters = {'a': (1, 2), 'b': (3., 3.5), 'c': 5}
    sample = ap.sample(parameters, n=2)
    assert sample == out


def test_rounded_sample():
    parameters = {'a': (1, 2), 'b': (3., 3.5), 'c': 5}
    sample = ap.sample(parameters, n=2, digits=0)
    out[1]['b'] = out[3]['b'] = 4
    assert sample == out


def test_sample_discrete():
    parameters = {'a': (1, 2), 'b': (3., 4.), 'c': 5}
    sample = ap.sample_discrete(parameters)
    assert sample == out


def test_sample_saltelli():
    parameters = {'a': (1, 2), 'b': (3., 4.), 'c': 5}
    sample = ap.sample_saltelli(parameters, n=1, digits=2)

    problem = {
        'num_vars': 2,
        'names': ['a', 'b', 'x3'],
        'bounds': [[1., 2.],
                   [3., 4.]]
    }
    param_values = saltelli.sample(problem, 1, calc_second_order=False)

    for s1, s2 in zip(sample, param_values):
        assert s1['a'] == int(round(s2[0]))
        assert s1['b'] == round(s2[1], 2)
        assert s1['c'] == 5
