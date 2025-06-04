from __future__ import annotations

from abc import ABCMeta
from typing import Callable

import numpy as np
from easyscience.Objects.core import ComponentSerializer
from easyscience.Objects.Inferface import ItemContainer

#if TYPE_CHECKING:
from easyreflectometry.model import Model
from easyreflectometry.sample import BaseAssembly
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer

from .wrapper_base import WrapperBase


class CalculatorBase(ComponentSerializer, metaclass=ABCMeta):
    """
    This class is a template and defines all properties that a calculator should have.
    """

    _calculators: list[CalculatorBase] = []  # class variable to store all calculators
    _material_link: dict[str, str]
    _layer_link: dict[str, str]
    _item_link: dict[str, str]
    _model_link: dict[str, str]

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs) -> None:
        r"""Initialise all subclasses so that they can be created in the factory

        :param is_abstract: Is this a subclass which shouldn't be dded
        :param kwargs: key word arguments
        """
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._calculators.append(cls)

    def __init__(self):
        self._namespace = {}
        self._wrapper: WrapperBase

    def reset_storage(self) -> None:
        """Reset the storage area of the calculator"""
        self._wrapper.reset_storage()

    def create(self, model: Material | Layer | Multilayer | Model) -> list[ItemContainer]:
        """Creation function

        :param model: Object to be created
        """
        r_list = []
        t_ = type(model)
        if issubclass(t_, Material):
            key = model.unique_name
            if key not in self._wrapper.storage['material'].keys():
                self._wrapper.create_material(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._material_link,
                    self._wrapper.get_material_value,
                    self._wrapper.update_material,
                )
            )
        elif issubclass(t_, MaterialMixture):
            key = model.unique_name
            if key not in self._wrapper.storage['material'].keys():
                self._wrapper.create_material(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._material_link,
                    self._wrapper.get_material_value,
                    self._wrapper.update_material,
                )
            )
        elif issubclass(t_, Layer):
            key = model.unique_name
            if key not in self._wrapper.storage['layer'].keys():
                self._wrapper.create_layer(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._layer_link,
                    self._wrapper.get_layer_value,
                    self._wrapper.update_layer,
                )
            )
            self.assign_material_to_layer(model.material.unique_name, key)
        elif issubclass(t_, BaseAssembly):
            key = model.unique_name
            self._wrapper.create_item(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._item_link,
                    self._wrapper.get_item_value,
                    self._wrapper.update_item,
                )
            )
            for i in model.layers:
                self.add_layer_to_item(i.unique_name, model.unique_name)
        elif issubclass(t_, Model):
            key = model.unique_name
            self._wrapper.create_model(key)
            r_list.append(
                ItemContainer(
                    key,
                    self._model_link,
                    self._wrapper.get_model_value,
                    self._wrapper.update_model,
                )
            )
            for i in model.sample:
                self.add_item_to_model(i.unique_name, key)
        return r_list

    def assign_material_to_layer(self, material_id: str, layer_id: str) -> None:
        """Assign a material to a layer.

        :param material_id: The material name
        :param layer_id: The layer name
        """
        self._wrapper.assign_material_to_layer(material_id, layer_id)

    def add_layer_to_item(self, layer_id: str, item_id: str) -> None:
        """Add a layer to the item stack

        :param item_id: The item id
        :param layer_id: The layer id
        """
        self._wrapper.add_layer_to_item(layer_id, item_id)

    def remove_layer_from_item(self, layer_id: str, item_id: str) -> None:
        """Remove a layer from an item stack

        :param item_id: The item id
        :param layer_id: The layer id
        """
        self._wrapper.remove_layer_from_item(layer_id, item_id)

    def add_item_to_model(self, item_id: str, model_id: str) -> None:
        """Add a layer to the item stack

        :param item_id: The item id
        :param model_id: The model id
        """
        self._wrapper.add_item(item_id, model_id)

    def remove_item_from_model(self, item_id: str, model_id: str) -> None:
        """Remove an item from the model

        :param item_id: The item id
        :param model_id: The model id
        """
        self._wrapper.remove_item(item_id, model_id)

    def reflectity_profile(self, x_array: np.ndarray, model_id: str) -> np.ndarray:
        """Determines the reflectivity profile for the given range and model.

        :param x_array: points to be calculated at
        :param model_id: The model id
        """
        return self._wrapper.calculate(x_array, model_id)

    def sld_profile(self, model_id: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the scattering length density profile.

        :param model_id: The model id
        :return: z and sld(z)
        """
        return self._wrapper.sld_profile(model_id)

    def set_resolution_function(self, resolution_function: Callable[[np.array], np.array]) -> None:
        return self._wrapper.set_resolution_function(resolution_function)

    @property
    def include_magnetism(self):
        return self._wrapper.magnetism

    @include_magnetism.setter
    def include_magnetism(self, magnetism: bool):
        """
        Set the magnetism flag for the calculator

        :param magnetism: True if the calculator should include magnetism
        """
        self._wrapper.magnetism = magnetism
