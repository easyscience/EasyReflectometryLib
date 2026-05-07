# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/arm61'

import matplotlib.pyplot as plt
import scipp as sc
from matplotlib.gridspec import GridSpec

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot(data: sc.DataGroup) -> None:
    """A general plotting function for easyreflectometry.

    Parameters
    ----------
    data : sc.DataGroup
        The DataGroup to be plotted.
    """
    if len([i for i in list(data.keys()) if 'SLD' in i]) == 0:
        plot_sld = False
        fig = plt.figure(figsize=(5, 3))
        gs = GridSpec(1, 1, figure=fig)
    else:
        plot_sld = True
        fig = plt.figure(figsize=(5, 6))
        gs = GridSpec(2, 1, figure=fig)
        ax2 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[0, 0])
    refl_nums = [k[3:] for k in data['coords'].keys() if 'Qz' == k[:2]]
    for i, refl_num in enumerate(refl_nums):
        plot_data = sc.DataArray(
            name=f'R_{refl_num}',
            data=data['data'][f'R_{refl_num}'].copy(),
            coords={f'Qz_{refl_num}': data['coords'][f'Qz_{refl_num}'].copy()},
        )
        plot_data.data *= sc.scalar(10.0**i, unit=plot_data.unit)
        plot_data.coords[f'Qz_{refl_num}'].variances = None
        sc.plot(plot_data, ax=ax1, norm='log', linestyle='', marker='.', color=color_cycle[i])
        try:
            plot_model_data = sc.DataArray(
                name=f'R_{refl_num}_model',
                data=data[f'R_{refl_num}_model'].copy(),
                coords={f'Qz_{refl_num}': data['coords'][f'Qz_{refl_num}'].copy()},
            )
            plot_model_data.data *= sc.scalar(10.0**i, unit=plot_model_data.unit)
            plot_model_data.coords[f'Qz_{refl_num}'].variances = None
            sc.plot(
                plot_model_data,
                ax=ax1,
                norm='log',
                linestyle='--',
                color=color_cycle[i],
                marker='',
            )
        except KeyError:
            pass
    ax1.autoscale(True)
    ax1.relim()
    ax1.autoscale_view()

    if plot_sld:
        for i, refl_num in enumerate(refl_nums):
            plot_sld_data = sc.DataArray(
                name=f'SLD_{refl_num}',
                data=data[f'SLD_{refl_num}'].copy(),
                coords={f'z_{refl_num}': data['coords'][f'z_{refl_num}'].copy()},
            )
            sc.plot(plot_sld_data, ax=ax2, linestyle='-', color=color_cycle[i], marker='')
        ax2.autoscale(True)
        ax2.relim()
        ax2.autoscale_view()
