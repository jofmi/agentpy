import pytest
import agentpy as ap
from SALib.sample import saltelli
from agentpy.tools import AgentpyError


def test_repr():
    v = ap.Values(1, 2)
    assert v.__repr__() == "Set of 2 parameter values"
    r = ap.Range(1, 2)
    assert r.__repr__() == "Parameter range from 1 to 2"
    ir = ap.IntRange(1, 2)
    assert ir.__repr__() == "Integer parameter range from 1 to 2"
    s = ap.Sample({'x': r}, 10)
    assert s.__repr__() == "Sample of 10 parameter combinations"


def test_seed():
    parameters = {'x': ap.Range(), 'seed': 1}
    sample = ap.Sample(parameters, 2, randomize=True)

    assert list(sample) == [{'x': 0.0, 'seed': 272996653310673477252411125948039410165},
                            {'x': 1.0, 'seed': 40125655066622386354123033417875897284}]

    sample = ap.Sample(parameters, 2, randomize=False)

    assert list(sample) == [{'x': 0.0, 'seed': 1},
                            {'x': 1.0, 'seed': 1}]


def test_errors():
    parameters = {'x': ap.Range(), 'seed': 1}
    with pytest.raises(AgentpyError):
        sample = ap.Sample(parameters)


def test_linspace_product():
    parameters = {
        'a': ap.IntRange(1, 2),
        'b': ap.Range(3, 3.5),
        'c': ap.Values(*'xyz'),
        'd': True
    }
    sample = ap.Sample(parameters, n=3)
    assert list(sample) == [
        {'a': 1, 'b': 3.0, 'c': 'x', 'd': True},
        {'a': 1, 'b': 3.0, 'c': 'y', 'd': True},
        {'a': 1, 'b': 3.0, 'c': 'z', 'd': True},
        {'a': 1, 'b': 3.25, 'c': 'x', 'd': True},
        {'a': 1, 'b': 3.25, 'c': 'y', 'd': True},
        {'a': 1, 'b': 3.25, 'c': 'z', 'd': True},
        {'a': 1, 'b': 3.5, 'c': 'x', 'd': True},
        {'a': 1, 'b': 3.5, 'c': 'y', 'd': True},
        {'a': 1, 'b': 3.5, 'c': 'z', 'd': True},
        {'a': 2, 'b': 3.0, 'c': 'x', 'd': True},
        {'a': 2, 'b': 3.0, 'c': 'y', 'd': True},
        {'a': 2, 'b': 3.0, 'c': 'z', 'd': True},
        {'a': 2, 'b': 3.25, 'c': 'x', 'd': True},
        {'a': 2, 'b': 3.25, 'c': 'y', 'd': True},
        {'a': 2, 'b': 3.25, 'c': 'z', 'd': True},
        {'a': 2, 'b': 3.5, 'c': 'x', 'd': True},
        {'a': 2, 'b': 3.5, 'c': 'y', 'd': True},
        {'a': 2, 'b': 3.5, 'c': 'z', 'd': True},
        {'a': 2, 'b': 3.0, 'c': 'x', 'd': True},
        {'a': 2, 'b': 3.0, 'c': 'y', 'd': True},
        {'a': 2, 'b': 3.0, 'c': 'z', 'd': True},
        {'a': 2, 'b': 3.25, 'c': 'x', 'd': True},
        {'a': 2, 'b': 3.25, 'c': 'y', 'd': True},
        {'a': 2, 'b': 3.25, 'c': 'z', 'd': True},
        {'a': 2, 'b': 3.5, 'c': 'x', 'd': True},
        {'a': 2, 'b': 3.5, 'c': 'y', 'd': True},
        {'a': 2, 'b': 3.5, 'c': 'z', 'd': True}
    ]


def test_linspace_zip():
    parameters = {
        'a': ap.Range(),  # 0, 1 Default
        'b': ap.Values(3, 4),
    }
    sample = ap.Sample(parameters, n=2, product=False)
    assert list(sample) == [{'a': 0.0, 'b': 3}, {'a': 1.0, 'b': 4}]


def test_sample_saltelli():
    parameters = {
        'a': ap.IntRange(1, 2),
        'b': ap.Range(3, 3.5),
        'c': ap.Values(*'xyz'),
        'd': True
    }
    sample = ap.Sample(parameters, n=2, method='saltelli', calc_second_order=False)

    problem = {
        'num_vars': 3,
        'names': ['a', 'b', 'c'],
        'bounds': [[1., 3.],
                   [3., 3.5],
                   [0., 3.],
                  ]
    }
    param_values = saltelli.sample(problem, 2, calc_second_order=False)

    for s1, s2 in zip(sample, param_values):
        assert s1['a'] == int(s2[0])
        assert s1['b'] == s2[1]
        assert s1['c'] == parameters['c'].values[int(s2[2])]
        assert s1['d'] == parameters['d']


def test_sample_saltelli_second():
    parameters = {
        'a': ap.IntRange(1, 2),
        'b': ap.Range(3, 3.5),
        'c': ap.Values(*'xyz'),
        'd': True
    }
    sample = ap.Sample(parameters, n=2, method='saltelli',
                       calc_second_order=True)

    problem = {
        'num_vars': 3,
        'names': ['a', 'b', 'c'],
        'bounds': [[1., 3.],
                   [3., 3.5],
                   [0., 3.],
                  ]
    }
    param_values = saltelli.sample(problem, 2, calc_second_order=True)

    for s1, s2 in zip(sample, param_values):
        assert s1['a'] == int(s2[0])
        assert s1['b'] == s2[1]
        assert s1['c'] == parameters['c'].values[int(s2[2])]
        assert s1['d'] == parameters['d']
