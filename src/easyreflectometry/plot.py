__author__ = 'github.com/arm61'

from itertools import cycle
from typing import Optional

import matplotlib.pyplot as plt
import scipp as sc
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot(data: sc.DataGroup) -> None:
    """
    A general plotting function for easyreflectometry.

    :param data: the DataGroup to be plotted.
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
            sc.plot(plot_model_data, ax=ax1, norm='log', linestyle='--', color=color_cycle[i], marker='')
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


def plot_sample_structure(sample, ax: Optional[Axes] = None) -> Axes:
    """Visualise Sample object as stacked layers, using matplotlib.

    Each layer is rendered as a rectangle with a height equal to its thickness. The plot starts at
    the sample's back layer (subphase) and stacks layers upwards towards the superphase, which is
    annotated as an infinite medium and therefore not drawn as a finite-height rectangle.

    :param sample: Sample to visualise.
    :param ax: Optional matplotlib axes to draw on. When ``None`` a new figure and axes are created.
    :returns: The axes containing the layer drawing.
    :raises TypeError: If *sample* is not a :class:`Sample` instance.
    :raises ValueError: If the sample does not contain any layers to plot.
    """

    # Import locally to avoid circular imports at module import time.
    from easyreflectometry.sample.collections.sample import Sample

    if not isinstance(sample, Sample):
        raise TypeError('sample must be an easyreflectometry.sample.Sample instance')

    assemblies = list(sample.data)
    if len(assemblies) == 0:
        raise ValueError('sample contains no assemblies to visualise')

    layers = []
    for assembly in assemblies:
        assembly_layers = getattr(assembly, 'layers', [])
        layers.extend([layer for layer in assembly_layers if layer is not None])

    if len(layers) == 0:
        raise ValueError('sample assemblies do not contain any layers to visualise')

    superphase_layer = getattr(sample, 'superphase', None)
    layers_bottom_to_top = [layer for layer in reversed(layers)]

    # Skip the superphase when drawing finite thickness rectangles.
    layers_to_plot = [layer for layer in layers_bottom_to_top if layer is not superphase_layer]

    if len(layers_to_plot) == 0:
        raise ValueError('sample does not contain finite layers to visualise')

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    width = 1.0
    current_height = 0.0
    color_cycler = cycle(color_cycle)
    text_kwargs = {'ha': 'center', 'fontsize': 9, 'color': 'black'}

    for index, layer in enumerate(layers_to_plot):
        color = next(color_cycler)
        thickness = float(getattr(layer.thickness, 'value', 0.0))

        label_prefix = 'Back layer: ' if index == 0 else ''
        label = f"{label_prefix}{layer.name}"

        if thickness <= 0:
            ax.hlines(current_height, 0, width, colors=color, linewidth=2)
            ax.text(width / 2, current_height, f'{label}\n0 Å', va='bottom', **text_kwargs)
            continue

        rect = Rectangle((0, current_height), width, thickness, facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(
            width / 2,
            current_height + thickness / 2,
            f'{label}\n{thickness:.1f} Å',
            va='center',
            **text_kwargs,
        )
        current_height += thickness

    top_margin = max(current_height * 0.05, 10.0)
    ax.set_xlim(0, width)
    ax.set_ylim(0, current_height + top_margin)
    ax.set_xticks([])
    ax.set_ylabel('Distance from substrate (Å)')
    title = getattr(sample, 'name', '') or 'Sample structure'
    ax.set_title(title)

    if superphase_layer is not None:
        ax.text(
            width / 2,
            current_height + top_margin * 0.5,
            f'{superphase_layer.name} (superphase)',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='black',
        )

    return ax
