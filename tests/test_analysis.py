import pytest
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt


def test_gridplot():
    """Test only for errors."""

    # Use cmap values
    grid1 = [[1, 1], [1, np.nan]]
    ap.gridplot(grid1)

    # Use RGB values
    x = (1., 1., 0.)
    grid2 = [[x, x], [x, x]]
    ap.gridplot(grid2)

    # Use color dict and convert
    cdict = {1: 'g'}
    ap.gridplot(grid1, color_dict=cdict, convert=True)

    # Use only convert
    grid3 = [['g', 'g'], ['g', np.nan]]
    ap.gridplot(grid3, convert=True)

    # Assign to axis
    fig, ax = plt.subplots()
    ap.gridplot(grid1, ax=ax)

    assert True


def test_animation():
    """Test only for errors."""
    def my_plot(model, ax):
        ax.set_title(f"{model.t}")

    fig, ax = plt.subplots()
    my_model = ap.Model({'steps': 2})
    animation = ap.animate(my_model, fig, ax, my_plot, skip=1)
    animation.to_jshtml()

    # Stop immediately
    my_model = ap.Model({'steps': 0})
    animation = ap.animate(my_model, fig, ax, my_plot)
    animation.to_jshtml()

    # Skip more than steps
    my_model = ap.Model({'steps': 0})
    animation = ap.animate(my_model, fig, ax, my_plot, skip=1)
    animation.to_jshtml()

    assert True


class MyModel(ap.Model):
    def step(self):
        self.measure('x', self.p.x)
        self.stop()


def test_sobol():
    si = 0.6593259637723373

    param_ranges = {'x': (0., 1.)}
    sample = ap.sample_saltelli(param_ranges, n=10)
    results = ap.Experiment(MyModel, sample).run()
    ap.sensitivity_sobol(results, param_ranges, measures='x')
    assert results.sensitivity['S1'][0] == si

    # Test if a non-varied parameter causes errors
    param_ranges = {'x': (0., 1.), 'y': 1}
    sample = ap.sample_saltelli(param_ranges, n=10)
    results = ap.Experiment(MyModel, sample).run()
    ap.sensitivity_sobol(results, param_ranges)
    assert results.sensitivity['S1'][0] == si

    # Test calc_second_order
    param_ranges = {'x': (0., 1.), 'y': 1}
    sample = ap.sample_saltelli(param_ranges, n=10, calc_second_order=True)
    results = ap.Experiment(MyModel, sample).run()
    ap.sensitivity_sobol(results, param_ranges, calc_second_order=True)
    assert results.sensitivity[('S2', 'x')][0].__repr__() == 'nan'
