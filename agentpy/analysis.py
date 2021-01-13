"""
Agentpy Analysis Module
Content: Sensitivity and interactive analysis, animation, visualization
"""

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets
import IPython

from SALib.analyze import sobol

from .tools import make_list, param_tuples_to_salib
from .objects import AgentList


def sensitivity_sobol(output, param_ranges, measures=None, **kwargs):
    """ Calculates Sobol Sensitivity Indices and adds them to the output,
    using :func:`SALib.analyze.sobol.analyze`.

    Arguments:
        output (DataDict): The output of an experiment that was set to only
            one iteration (default) and used a parameter sample that was
            generated with :func:`sample_saltelli`.
        param_ranges (dict): The same dictionary that was used for the
            generation of the parameter sample with :func:`sample_saltelli`.
        measures (str or list of str, optional): The measures that should
            be used for the analysis. If none are passed, all are used.
        **kwargs: Will be forwarded to :func:`SALib.analyze.sobol.analyze`.
            The kwarg ``calc_second_order`` must be the same as for
            :func:`sample_saltelli`.
    """

    # STEP 1 - Convert param_ranges to SALib Format
    param_ranges_tuples = {k: v for k, v in param_ranges.items()
                           if isinstance(v, tuple)}
    param_ranges_salib = param_tuples_to_salib(param_ranges_tuples)

    # STEP 2 - Calculate Sobol Sensitivity Indices

    if measures is None:
        measures = output.measures.columns

    if isinstance(measures, str):
        measures = make_list(measures)

    dfs_si = []
    dfs_si_conf = []
    dfs_list = [dfs_si, dfs_si_conf]

    for measure in measures:
        y = np.array(output.measures[measure])
        si = sobol.analyze(param_ranges_salib, y, **kwargs)

        # Make dataframes out of sensitivities
        keys_list = [['S1', 'ST'], ['S1_conf', 'ST_conf']]
        for dfs, keys in zip(dfs_list, keys_list):
            s = {k: v for k, v in si.items() if k in keys}
            df = pd.DataFrame(s)
            var_pars = output._combine_pars(varied=True, fixed=False)
            df['parameter'] = var_pars.keys()
            df['measure'] = measure
            df = df.set_index(['measure', 'parameter'])
            dfs.append(df)

    output['sensitivity'] = pd.concat(dfs_si)
    output['sensitivity_conf'] = pd.concat(dfs_si_conf)
    # TODO Second-Order Entries Missing

    return output


def animate(model, fig, axs, plot, steps=None,
            skip=0, fargs=(), **kwargs):
    """ Returns an animation of the model simulation,
    using :func:`matplotlib.animation.FuncAnimation`.

    Arguments:
        model (Model): The model instance.
        fig (matplotlib.figure.Figure): Figure for the animation.
        axs (matplotlib.axes.Axes or list): Axis or list of axis of the figure.
        plot (function): Function that takes `(model, ax, *fargs)`
            and creates the desired plots on each axis at each time-step.
        steps(int, optional):
                Maximum number of steps for the simulation to run.
                If none is given, the parameter 'Model.p.steps' will be used.
                If there is no such parameter, 'steps' will be set to 1000.
        skip (int, optional): Number of rounds to skip before the
            animation starts (default 0).
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

    model._setup_run(steps)
    model._create_output()
    pre_steps = 0

    for _ in range(skip):
        model._make_step()

    def frames():
        nonlocal model, pre_steps
        if model._stop is False:
            while not model._stop:
                if pre_steps < 2:  # Frames iterates twice before starting plot
                    pre_steps += 1
                else:
                    model._make_step()
                    model._create_output()
                yield model.t
        else:  # Yield current if model stops before the animation starts
            yield model.t

    def update(t, m, axs, *fargs):  # noqa
        nonlocal pre_steps
        for ax in make_list(axs):
            # Clear axes before each plot
            ax.clear()
        plot(m, axs, *fargs)  # Perform plot

    ani = matplotlib.animation.FuncAnimation(
        fig, update, frames=frames, fargs=(model, axs, *fargs), **kwargs)  # noqa

    plt.close()  # Don't display static plot
    return ani


def _apply_colors(grid, color_dict, convert):
    if isinstance(grid[0], list):
        return [_apply_colors(subgrid, color_dict, convert)
                for subgrid in grid]
    else:
        if color_dict is not None:
            grid = [i if i is np.nan else color_dict[i] for i in grid]
        if convert is True:
            grid = [(0., 0., 0., 0.) if i is np.nan else
                    matplotlib.colors.to_rgba(i) for i in grid]
        return grid


def gridplot(grid, color_dict=None, convert=False, ax=None, **kwargs):
    """ Visualizes values on a two-dimensional grid with
    :func:`matplotlib.pyplot.imshow`.

    Arguments:
        grid(list of list): Two-dimensional grid with values.
            numpy.nan values will be plotted as empty patches.
        color_dict(dict, optional): Dictionary that translates
            each value in `grid` to a color specification.
        convert(bool, optional): Convert values to rgba vectors,
             using :func:`matplotlib.colors.to_rgba` (default False).
        ax(matplotlib.pyplot.axis, optional): Axis to be used for plot.
        **kwargs: Forwarded to :func:`matplotlib.pyplot.imshow`.
     """

    # TODO Make feature for legend
    if color_dict is not None or convert:
        grid = _apply_colors(grid, color_dict, convert)
    if ax:
        ax.imshow(grid, **kwargs)
    else:
        plt.imshow(grid, **kwargs)
