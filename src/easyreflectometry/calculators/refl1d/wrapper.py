# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from typing import Tuple

import numpy as np
from refl1d import names
from refl1d.sample.layers import Repeat

from ..polarization import POLARIZATION_CHANNEL_TO_INDEX
from ..wrapper_base import WrapperBase

RESOLUTION_PADDING = 3.5
OVERSAMPLING_FACTOR = 21


class Refl1dWrapper(WrapperBase):
    supports_magnetism = True

    def create_material(self, name: str):
        """Create a material using SLD.

        Parameters
        ----------
        name : str
            The name of the material.
        """
        self.storage['material'][name] = names.SLD(str(name))

    def create_layer(self, name: str):
        """Create a layer using Slab.

        Parameters
        ----------
        name : str
            The name of the layer.
        """
        if self._magnetism:
            magnetism = names.Magnetism(rhoM=0.0, thetaM=0.0)
        else:
            magnetism = None
        self.storage['layer'][name] = names.Slab(name=str(name), magnetism=magnetism)

    def create_item(self, name: str):
        """Create an item using Repeat.

        Parameters
        ----------
        name : str
            The name of the item.
        """
        self.storage['item'][name] = Repeat(names.Stack(names.Slab(names.SLD(), thickness=0, interface=0)), name=str(name))
        del self.storage['item'][name].stack[0]

    def update_layer(self, name: str, **kwargs):
        """Update a layer in a given item.

        Parameters
        ----------
        name : str
            The layer name.
        **kwargs :
        """
        kwargs_no_magnetism = {k: v for k, v in kwargs.items() if k != 'magnetism_rhoM' and k != 'magnetism_thetaM'}
        super().update_layer(name, **kwargs_no_magnetism)
        if any(item.startswith('magnetism') for item in kwargs.keys()):
            magnetism = names.Magnetism(rhoM=kwargs['magnetism_rhoM'], thetaM=kwargs['magnetism_thetaM'])
            self.storage['layer'][name].magnetism = magnetism

    def get_layer_value(self, name: str, key: str) -> float:
        """A function to get a given layer value.

        Parameters
        ----------
        name : str
            The layer name.
        key : str
            The given value keys.
        """
        if key in ['magnetism_rhoM', 'magnetism_thetaM']:
            return getattr(
                self.storage['layer'][name].magnetism, key.split('_')[-1]
            ).value  # TODO: check if we want to return the raw value or the full Parameter  # noqa: E501
        return super().get_layer_value(name, key)

    def _remove_magnetism_from_layers(self) -> None:
        """Detach Magnetism objects from all slabs.

        Called when magnetism is disabled: slabs carrying Magnetism objects would
        crash refl1d's plain (unpolarized) QProbe path. Magnetic parameters must be
        set again (via `update_layer`) after re-enabling magnetism.
        """
        for layer in self.storage['layer'].values():
            layer.magnetism = None

    def create_model(self, name: str):
        """Create a model for analysis.

        Parameters
        ----------
        name : str
            Name for the model.
        """
        self.storage['model'][name] = {'scale': 1, 'bkg': 0, 'items': []}

    def update_model(self, name: str, **kwargs):
        """Update the non-structural parameters of the model.

        Parameters
        ----------
        **kwargs :
        name : str
            Name of the model.
        """
        model = self.storage['model'][name]
        for key in kwargs.keys():
            model[key] = kwargs[key]

    def get_model_value(self, name: str, key: str) -> float:
        """A function to get a given model value.

        Parameters
        ----------
        name : str
            Name of the model.
        key : str
            The given value keys.

        Returns
        -------
        float
            The desired value.
        """
        model = self.storage['model'][name]
        return model[key]

    def assign_material_to_layer(self, material_name: str, layer_name: str):
        """Assign a material to a layer.

        Parameters
        ----------
        material_name : str
            The material name.
        layer_name : str
            The layer name.
        """
        self.storage['layer'][layer_name].material = self.storage['material'][material_name]

    def add_layer_to_item(self, layer_name: str, item_name: str):
        """Create a layer from the material of the same name, in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        item = self.storage['item'][item_name]
        item.stack.add(self.storage['layer'][layer_name])

    def add_item(self, item_name: str, model_name: str):
        """Add an item to the model.

        Parameters
        ----------
        item_name : str
            Items to add to model.
        model_name : str
            Name for the model.
        """
        self.storage['model'][model_name]['items'].append(self.storage['item'][item_name])

    def remove_layer_from_item(self, layer_name: str, item_name: str):
        """Remove a layer in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        layer_idx = list(self.storage['item'][item_name].stack).index(self.storage['layer'][layer_name])
        del self.storage['item'][item_name].stack[layer_idx]

    def remove_item(self, item_name: str, model_name: str):
        """Remove a given item.

        Parameters
        ----------
        item_name : str
            The item name.
        model_name : str
            The model name.
        """
        item_idx = self.storage['model'][model_name]['items'].index(self.storage['item'][item_name])
        del self.storage['model'][model_name]['items'][item_idx]
        del self.storage['item'][item_name]

    def calculate(self, q_array: np.ndarray, model_name: str) -> np.ndarray:
        """For a given q array calculate the corresponding reflectivity.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_name : str
            The model name.

        Returns
        -------
        np.ndarray
            Reflectivity calculated at q.
        """
        if self._magnetism:
            reflectivities = self._polarized_reflectivities(q_array, model_name)
            return reflectivities[POLARIZATION_CHANNEL_TO_INDEX[self._polarization_channel]]

        sample = _build_sample(self.storage, model_name)
        # smearing() returns sigma, which is exactly what refl1d's probe.dQ expects.
        dq_array = self._resolution_function.smearing(q_array)
        probe = _get_probe(
            q_array=q_array,
            dq_array=dq_array,
            model_name=model_name,
            storage=self.storage,
            oversampling_factor=OVERSAMPLING_FACTOR,
        )
        # returns q, reflectivity
        _, reflectivity = names.Experiment(probe=probe, sample=sample).reflectivity()
        return reflectivity

    def calculate_polarized(self, q_array: np.ndarray, model_name: str) -> dict[str, np.ndarray]:
        """For a given q array calculate the reflectivity of all four spin channels.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_name : str
            The model name.

        Returns
        -------
        dict[str, np.ndarray]
            Reflectivity per spin channel, keyed 'pp', 'pm', 'mp', 'mm' (in that order).
        """
        if not self._magnetism:
            raise ValueError(
                'Polarized reflectivity requires magnetism: enable it on this calculator first '
                '(`include_magnetism = True` on the calculator / `magnetism = True` on the wrapper).'
            )
        reflectivities = self._polarized_reflectivities(q_array, model_name)
        return {channel.value: reflectivities[index] for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items()}

    def _polarized_reflectivities(self, q_array: np.ndarray, model_name: str) -> list:
        """Reflectivity of the four spin cross-sections, in refl1d order (pp, pm, mp, mm)."""
        sample = _build_sample(self.storage, model_name)
        dq_array = self._resolution_function.smearing(q_array)
        polarized_probe = _get_polarized_probe(
            q_array=q_array,
            dq_array=dq_array,
            model_name=model_name,
            storage=self.storage,
            oversampling_factor=OVERSAMPLING_FACTOR,
        )
        polarized_reflectivity = names.Experiment(probe=polarized_probe, sample=sample).reflectivity()

        # returns (q, reflectivity) per cross-section
        reflectivities = [reflectivity for _, reflectivity in polarized_reflectivity]
        if len(reflectivities) != 4:
            raise RuntimeError(f'refl1d returned {len(reflectivities)} polarized cross-sections; expected 4.')
        for channel, index in POLARIZATION_CHANNEL_TO_INDEX.items():
            if len(reflectivities[index]) != len(q_array) or not np.all(np.isfinite(reflectivities[index])):
                raise RuntimeError(f'refl1d returned a malformed {channel.value} cross-section.')
        return reflectivities

    def sld_profile(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_name : str
            The model name.

        Returns
        -------

            Z and sld(z).
        """
        sample = _build_sample(self.storage, model_name)
        probe = _get_probe(
            q_array=np.array([1]),  # dummy value
            dq_array=np.array([1]),  # dummy value
            model_name=model_name,
            storage=self.storage,
        )
        z, sld, _ = names.Experiment(probe=probe, sample=sample).smooth_profile()
        # -1 to reverse the order
        return z, sld[::-1]

    def magnetic_sld_profile(self, model_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the nuclear and magnetic scattering length density profiles.

        Parameters
        ----------
        model_name : str
            The model name.

        Returns
        -------

            z, sld(z), magnetic sld rhoM(z) and magnetic angle thetaM(z).
        """
        if not self._magnetism:
            raise ValueError(
                'The magnetic sld profile requires magnetism: enable it on this calculator first '
                '(`include_magnetism = True` on the calculator / `magnetism = True` on the wrapper).'
            )
        sample = _build_sample(self.storage, model_name)
        probe = _get_probe(
            q_array=np.array([1]),  # dummy value
            dq_array=np.array([1]),  # dummy value
            model_name=model_name,
            storage=self.storage,
        )
        z, sld, _, sld_magnetic, theta_magnetic = names.Experiment(probe=probe, sample=sample).magnetic_smooth_profile()
        # -1 to reverse the order
        return z, sld[::-1], sld_magnetic[::-1], theta_magnetic[::-1]


def _get_oversampling_q(q_array: np.ndarray, dq_array: np.ndarray, oversampling_factor: int) -> np.ndarray:
    """Get oversampling q."""
    argmin = np.argmin(q_array)  # index of the smallest q element
    argmax = np.argmax(q_array)  # index of the largest q element
    return np.linspace(
        q_array[argmin] - RESOLUTION_PADDING * dq_array[argmin],  # dq element at the smallest q index
        q_array[argmax] + RESOLUTION_PADDING * dq_array[argmax],  # dq element at the largest q index
        oversampling_factor * len(q_array),
    )


def _get_probe(
    q_array: np.ndarray,
    dq_array: np.ndarray,
    model_name: str,
    storage: dict,
    oversampling_factor: int = 1,
    magnetism: bool = False,
) -> names.QProbe:
    """Get probe."""
    probe = names.QProbe(
        Q=q_array,
        dQ=dq_array,
        intensity=storage['model'][model_name]['scale'],
        background=storage['model'][model_name]['bkg'],
    )

    # Add theta_offset attribute if magnetism is enabled
    # This is required for PolarizedQProbe to work correctly
    if magnetism:
        probe.theta_offset = names.Parameter.default(0, name='theta_offset')

    if oversampling_factor > 1:
        probe.calc_Qo = _get_oversampling_q(q_array, dq_array, oversampling_factor)
    return probe


def _get_polarized_probe(
    q_array: np.ndarray,
    dq_array: np.ndarray,
    model_name: str,
    storage: dict,
    oversampling_factor: int = 1,
) -> names.PolarizedNeutronQProbe:
    """Get polarized probe with all four cross-sections (pp, pm, mp, mm)."""
    four_probes = [
        _get_probe(
            q_array=q_array,
            dq_array=dq_array,
            model_name=model_name,
            storage=storage,
            oversampling_factor=oversampling_factor,
            magnetism=True,  # Enable magnetism for polarized probes
        )
        for _ in range(4)
    ]

    # Create polarized probe and work around initialization bug
    polarized_probe = names.PolarizedNeutronQProbe.__new__(names.PolarizedNeutronQProbe)
    polarized_probe._union_cache_key = None  # Initialize missing attribute
    polarized_probe.__init__(xs=four_probes, name='polarized')
    return polarized_probe


def _build_sample(storage: dict, model_name: str) -> names.Stack:
    """Build sample."""
    sample = names.Stack()
    # -1 to reverse the order
    for i in storage['model'][model_name]['items'][::-1]:
        if i.repeat.value == 1:
            # -1 to reverse the order
            for j in range(len(i.stack))[::-1]:
                sample |= i.stack[j]
        else:
            stack = names.Stack()
            # -1 to reverse the order
            for j in range(len(i.stack))[::-1]:
                stack |= i.stack[j]
            sample |= Repeat(stack, repeat=i.repeat.value)
    return sample
