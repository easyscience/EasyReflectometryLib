__author__ = 'github.com/arm61'

from typing import Tuple

import numpy as np
from refl1d import model
from refl1d import names

from easyreflectometry.model import PercentageFwhm

from ..wrapper_base import WrapperBase

RESOLUTION_PADDING = 3.5
OVERSAMPLING_FACTOR = 21
ALL_POLARIZATIONS = False


class Refl1dWrapper(WrapperBase):
    def create_material(self, name: str):
        """
        Create a material using SLD.

        :param name: The name of the material
        """
        self.storage['material'][name] = names.SLD(str(name))

    def create_layer(self, name: str):
        """
        Create a layer using Slab.

        :param name: The name of the layer
        """
        if self._magnetism:
            magnetism = names.Magnetism(rhoM=0.0, thetaM=0.0)
        else:
            magnetism = None
        self.storage['layer'][name] = model.Slab(name=str(name), magnetism=magnetism)

    def create_item(self, name: str):
        """
        Create an item using Repeat.

        :param name: The name of the item
        """
        self.storage['item'][name] = model.Repeat(
            model.Stack(model.Slab(names.SLD(), thickness=0, interface=0)), name=str(name)
        )
        del self.storage['item'][name].stack[0]

    def update_layer(self, name: str, **kwargs):
        """Update a layer in a given item.

        :param name: The layer name.
        :param kwargs:
        """
        kwargs_no_magnetism = {k: v for k, v in kwargs.items() if k != 'magnetism_rhoM' and k != 'magnetism_thetaM'}
        super().update_layer(name, **kwargs_no_magnetism)
        if any(item.startswith('magnetism') for item in kwargs.keys()):
            magnetism = names.Magnetism(rhoM=kwargs['magnetism_rhoM'], thetaM=kwargs['magnetism_thetaM'])
            self.storage['layer'][name].magnetism = magnetism

    def get_layer_value(self, name: str, key: str) -> float:
        """A function to get a given layer value

        :param name: The layer name
        :param key: The given value keys
        """
        if key in ['magnetism_rhoM', 'magnetism_thetaM']:
            return getattr(self.storage['layer'][name].magnetism, key.split('_')[-1])
        return super().get_layer_value(name, key)

    def create_model(self, name: str):
        """
        Create a model for analysis

        :param name: Name for the model
        """
        self.storage['model'][name] = {'scale': 1, 'bkg': 0, 'items': []}

    def update_model(self, name: str, **kwargs):
        """
        Update the non-structural parameters of the model

        :param name: Name of the model
        """
        model = self.storage['model'][name]
        for key in kwargs.keys():
            model[key] = kwargs[key]

    def get_model_value(self, name: str, key: str) -> float:
        """
        A function to get a given model value

        :param name: Name of the model
        :param key: The given value keys
        :return: The desired value
        """
        model = self.storage['model'][name]
        return model[key]

    def assign_material_to_layer(self, material_name: str, layer_name: str):
        """
        Assign a material to a layer.

        :param material_name: The material name
        :param layer_name: The layer name
        """
        self.storage['layer'][layer_name].material = self.storage['material'][material_name]

    def add_layer_to_item(self, layer_name: str, item_name: str):
        """
        Create a layer from the material of the same name, in a given item.

        :param layer_name: The layer name
        :param item_name: The item name
        """
        item = self.storage['item'][item_name]
        item.stack.add(self.storage['layer'][layer_name])

    def add_item(self, item_name: str, model_name: str):
        """
        Add an item to the model.

        :param item_name: items to add to model
        :param model_name: name for the model
        """
        self.storage['model'][model_name]['items'].append(self.storage['item'][item_name])

    def remove_layer_from_item(self, layer_name: str, item_name: str):
        """
        Remove a layer in a given item.

        :param layer_name: The layer name
        :param item_name: The item name
        """
        layer_idx = list(self.storage['item'][item_name].stack).index(self.storage['layer'][layer_name])
        del self.storage['item'][item_name].stack[layer_idx]

    def remove_item(self, item_name: str, model_name: str):
        """
        Remove a given item.

        :param item_name: The item name
        :param model_name: The model name
        """
        item_idx = self.storage['model'][model_name]['items'].index(self.storage['item'][item_name])
        del self.storage['model'][model_name]['items'][item_idx]
        del self.storage['item'][item_name]

    def calculate(self, q_array: np.ndarray, model_name: str) -> np.ndarray:
        """For a given q array calculate the corresponding reflectivity.

        :param q_array: array of data points to be calculated
        :param model_name: the model name
        :return: reflectivity calculated at q
        """
        sample = _build_sample(self.storage, model_name)
        dq_array = self._resolution_function.smearing(q_array)

        if isinstance(self._resolution_function, PercentageFwhm):
            # Get percentage of Q and change from sigma to FWHM
            dq_array = dq_array * q_array / 100 / (2 * np.sqrt(2 * np.log(2)))

        if not self._magnetism:
            probe = _get_probe(
                q_array=q_array,
                dq_array=dq_array,
                model_name=model_name,
                storage=self.storage,
                oversampling_factor=OVERSAMPLING_FACTOR,
            )
            # returns q, reflectivity
            _, reflectivity = names.Experiment(probe=probe, sample=sample).reflectivity()
        else:
            polarized_probe = _get_polarized_probe(
                q_array=q_array,
                dq_array=dq_array,
                model_name=model_name,
                storage=self.storage,
                oversampling_factor=OVERSAMPLING_FACTOR,
                all_polarizations=ALL_POLARIZATIONS,
            )
            polarized_reflectivity = names.Experiment(probe=polarized_probe, sample=sample).reflectivity()

            if ALL_POLARIZATIONS:
                raise NotImplementedError('Polarized reflectivity not yet implemented')
                # returns q, reflectivity
                # _, reflectivity_pp = polarized_reflectivity[0]
                # _, reflectivity_pm = polarized_reflectivity[1]
                # _, reflectivity_mp = polarized_reflectivity[2]
                # _, reflectivity_mm = polarized_reflectivity[3]
            else:
                # Only pick the pp reflectivity
                # returns q, reflectivity
                _, reflectivity = polarized_reflectivity[0]

        return reflectivity

    def sld_profile(self, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the scattering length density profile.

        :param model_name: the model name
        :return: z and sld(z)
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


def _get_oversampling_q(q_array: np.ndarray, dq_array: np.ndarray, oversampling_factor: int) -> np.ndarray:
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
) -> names.QProbe:
    probe = names.QProbe(
        Q=q_array,
        dQ=dq_array,
        intensity=storage['model'][model_name]['scale'],
        background=storage['model'][model_name]['bkg'],
    )
    if oversampling_factor > 1:
        probe.calc_Qo = _get_oversampling_q(q_array, dq_array, oversampling_factor)
    return probe


def _get_polarized_probe(
    q_array: np.ndarray,
    dq_array: np.ndarray,
    model_name: str,
    storage: dict,
    oversampling_factor: int = 1,
    all_polarizations: bool = False,
) -> names.PolarizedQProbe:
    four_probes = []
    for i in range(4):
        if i == 0 or all_polarizations:
            probe = _get_probe(
                q_array=q_array,
                dq_array=dq_array,
                model_name=model_name,
                storage=storage,
                oversampling_factor=oversampling_factor,
            )
        else:
            probe = None
        four_probes.append(probe)
    return names.PolarizedQProbe(xs=four_probes, name='polarized')


def _build_sample(storage: dict, model_name: str) -> model.Stack:
    sample = model.Stack()
    # -1 to reverse the order
    for i in storage['model'][model_name]['items'][::-1]:
        if i.repeat.value == 1:
            # -1 to reverse the order
            for j in range(len(i.stack))[::-1]:
                sample |= i.stack[j]
        else:
            stack = model.Stack()
            # -1 to reverse the order
            for j in range(len(i.stack))[::-1]:
                stack |= i.stack[j]
            sample |= model.Repeat(stack, repeat=i.repeat.value)
    return sample
