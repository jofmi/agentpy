import pytest
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt


def test_gridplot():
    """Test only for errors."""

    # Use cmap values
    grid1 = np.array([[1, 1], [1, np.nan]])
    ap.gridplot(grid1)

    # Use RGB values
    x = (1., 1., 0.)
    grid2 = np.array([[x, x], [x, x]])
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
