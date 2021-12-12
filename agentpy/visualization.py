"""
Agentpy Visualization Module
Content: Animations and Gridplot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
from matplotlib.animation import FuncAnimation
from SALib.analyze import sobol

from .tools import make_list, param_tuples_to_salib


def animate(model, fig, axs, plot, steps=None, seed=None,
            skip=0, fargs=(), **kwargs):
    """ Returns an animation of the model simulation,
    using :func:`matplotlib.animation.FuncAnimation`.

    Arguments:
        model (Model): The model instance.
        fig (matplotlib.figure.Figure): Figure for the animation.
        axs (matplotlib.axes.Axes or list): Axis or list of axis of the figure.
        plot (function): Function that takes the arguments `model, axs, *fargs`
            and creates the desired plots on each axis at each time-step.
        steps(int, optional):
            Number of (additional) steps for the simulation to run.
            If passed, the parameter 'Model.p.steps' will be ignored.
            The simulation can still be stopped with :func:'Model.stop'.
            If there is no step-limit through either this argument or
            the parameter 'Model.p.steps', the animation will stop at t=10000.
        seed (int, optional):
            Seed for the models random number generators.
            If none is given, the parameter 'Model.p.seed' will be used.
            If there is no such parameter, a random seed will be used.
        skip (int, optional):
            Steps to skip before the animation starts (default 0).
        fargs (tuple, optional): Forwarded fo the `plot` function.
        **kwargs: Forwarded to :func:`matplotlib.animation.FuncAnimation`.

    Examples:
        An animation can be generated as follows::

            def my_plot(model, ax):
                pass  # Call pyplot functions here
            
            fig, ax = plt.subplots() 
            my_model = MyModel(parameters)
            animation = ap.animate(my_model, fig, ax, my_plot)

        One way to display the resulting animation object in Jupyter::

            from IPython.display import HTML
            HTML(animation.to_jshtml())
    """

    model.sim_setup(steps, seed)
    model.create_output()
    pre_steps = 0

    for _ in range(skip):
        model.sim_step()

    def frames():
        nonlocal model, pre_steps
        if model.running is True:
            while model.running:
                if pre_steps < 2:  # Frames iterates twice before starting plot
                    pre_steps += 1
                else:
                    model.sim_step()
                    model.create_output()
                yield model.t
        else:  # Yield current if model stops before the animation starts
            yield model.t

    def update(t, m, axs, *fargs):  # noqa
        nonlocal pre_steps
        for ax in make_list(axs):
            # Clear axes before each plot
            ax.clear()
        plot(m, axs, *fargs)  # Perform plot

    save_count = 10000 if model._steps is np.nan else model._steps + 1

    ani = FuncAnimation(
        fig, update,
        frames=frames,
        fargs=(model, axs, *fargs),
        save_count=save_count,  # Limits animation to 100 steps otherwise
        **kwargs)  # noqa

    plt.close()  # Don't display static plot
    return ani


def _apply_colors(grid, color_dict, convert):
    if color_dict is not None:
        if None in color_dict:
            def func(v):
                return color_dict[None] if np.isnan(v) else color_dict[v]
        else:
            def func(v):
                return np.nan if np.isnan(v) else color_dict[v]
        grid = np.vectorize(func)(grid)
    if convert is True:
        def func(v):
            # TODO Can be improved
            if isinstance(v, str):
                if v == 'nan':
                    return 0., 0., 0., 0.
                else:
                    return to_rgba(v)
            elif np.isnan(v):
                return 0., 0., 0., 0.
            else:
                return to_rgba(v)
        grid = np.vectorize(func)(grid)
        grid = np.moveaxis(grid, 0, 2)
    return grid


def gridplot(grid, color_dict=None, convert=False, ax=None, **kwargs):
    """ Visualizes values on a two-dimensional grid with
    :func:`matplotlib.pyplot.imshow`.
    
    Arguments:
        grid (numpy.array): Two-dimensional array with values.
            numpy.nan values will be plotted as empty patches.
        color_dict (dict, optional): Dictionary that translates
            each value in `grid` to a color specification.
            If there is an entry `None`, it will be used for all NaN values.
        convert (bool, optional): Convert values to rgba vectors,
             using :func:`matplotlib.colors.to_rgba` (default False).
        ax (matplotlib.pyplot.axis, optional): Axis to be used for plot.
        **kwargs: Forwarded to :func:`matplotlib.pyplot.imshow`.
        
    Returns:
        :class:`matplotlib.image.AxesImage`  
    """
    # TODO Make feature for legend
    if color_dict is not None or convert:
        grid = _apply_colors(grid, color_dict, convert)
    if ax:
        im = ax.imshow(grid, **kwargs)
    else:
        im = plt.imshow(grid, **kwargs)
    return im
