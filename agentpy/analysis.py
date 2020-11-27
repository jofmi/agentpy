"""
Agentpy Analysis Module
Content: Sensitivity and interactive analysis, animation, visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ipywidgets as widgets

from scipy.interpolate import griddata
from SALib.analyze import sobol
from matplotlib import animation

from .tools import make_list, param_tuples_to_salib
from .framework import AgentList


def sobol_sensitivity(output, param_ranges, measures=None, **kwargs):
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
            df['parameter'] = output.parameters.varied.keys()
            df['measure'] = measure
            df = df.set_index(['measure', 'parameter'])
            dfs.append(df)

    output['sensitivity'] = pd.concat(dfs_si)
    output['sensitivity_conf'] = pd.concat(dfs_si_conf)

    # TODO Second-Order Entries Missing

    return output['sensitivity']


def animate(model, parameters, fig, axs, output_function,
            skip_t0=False, **kwargs):
    """ Returns an animation of the model simulation

    Arguments:
        model (type): The model class type.
        parameters (dict): Model parameters.
        fig: Figure for the animation.
        axs: Axis of the figure.
        output_function: Output to be displayed in the animation.
        skip_t0: Start animation at t == 1 (default False).
        **kwargs: Will be forwarded to output_function.

    Returns:
        ipywidgets.HBox: Interactive output widget
    """  # TODO Improve function & docs

    m = model(parameters)
    steps = m.p['steps'] if 'steps' in m.p else False
    m._stop = False
    m.setup()
    m.update()
    m._update_stop(steps)

    step0 = True
    step00 = not skip_t0

    def frames():

        nonlocal m, step0, step00

        while not m._stop:

            if step0:
                step0 = False
            elif step00:
                step00 = False
            else:
                m.t += 1
                m.step()
            m.update()
            m._update_stop(steps)
            m._create_output()
            yield m.t

    def update(t, m, axs):  # noqa

        for ax in make_list(axs):
            ax.clear()

        if m.t == 0 and skip_t0:
            pass
        else:
            output_function(m, axs)

    ani = animation.FuncAnimation(fig, update, frames=frames,  # noqa
                                  fargs=(m, axs), **kwargs)

    plt.close()  # Don't display static plot
    return ani


def interactive(model, param_ranges, output_function, *args, **kwargs):
    """
    Returns 'output_function' as an interactive ipywidget
    More infos at https://ipywidgets.readthedocs.io/

    Arguments:
        model (type): The model class type.
        param_ranges (dict): The parameter ranges to be tested.
        output_function (function): Output to be displayed interactively.
        *args: Will be forwarded to output_function.
        **kwargs: Will be forwarded to output_function.

    Returns:
        ipywidgets.HBox: Interactive output widget
    """

    def var_run(**param_updates):

        parameters = dict(param_ranges)
        parameters.update(param_updates)
        temp_model = model(parameters)
        temp_model.run(display=False)

        output_function(temp_model.output, *args, **kwargs)

    # Create widget dict
    widget_dict = {}
    param_ranges_tuples = {k: v for k, v in param_ranges.items()
                           if isinstance(v, tuple)}
    for var_key, var_range in param_ranges_tuples.items():
        widget_dict[var_key] = widgets.FloatSlider(
            description=var_key,
            value=(var_range[1] - var_range[0]) / 2,
            min=var_range[0],
            max=var_range[1],
            step=(var_range[1] - var_range[0]) / 10,
            style=dict(description_width='initial'),
            layout={'width': '300px'})

    out = widgets.interactive_output(var_run, widget_dict)

    return widgets.HBox([widgets.VBox(list(widget_dict.values())), out])


def gridplot(model, ax, grid_key, attr_key, color_dict):
    """ Plots a 2D agent grid """

    def apply_colors(grid, color_assignment, final_type, attr_key):
        if not isinstance(grid[0], final_type):
            return [apply_colors(subgrid, color_assignment, final_type,
                    attr_key) for subgrid in grid]
        else:
            return [colors.to_rgb(color_assignment(i, attr_key))
                    for i in grid]

    def color_assignment(a_list, attr_key):
        if len(a_list) == 0:
            return color_dict['empty']
        else:
            return color_dict[a_list[0][attr_key]]

    grid = model.envs[grid_key]._grid
    color_grid = apply_colors(grid, color_assignment, AgentList, attr_key)
    _ = ax.imshow(color_grid)


def phaseplot(data, x, y, z, n, fill=True, **kwargs):
    """ Creates a contour plot displaying the interpolated
    sensitivity between the parameters x,y and the measure z """
    # TODO Function unfinished

    # Create grid
    x_vals = np.linspace(min(data[x]), max(data[x]), n)
    y_vals = np.linspace(min(data[y]), max(data[y]), n)
    x, y = np.meshgrid(x_vals, y_vals)

    # Interpolate z
    z = griddata((data[x], data[y]), data[z], (x, y))

    # Create contour plot
    if fill:
        img = plt.contourf(x, y, z, **kwargs)
    else:
        img = plt.contour(x, y, z, **kwargs)

    # Create colorbar
    plt.colorbar(mappable=img)

    # Labels
    plt.title(z)
    plt.xlabel(x)
    plt.ylabel(y)

    plt.show()
