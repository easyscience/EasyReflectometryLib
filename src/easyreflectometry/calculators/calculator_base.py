"""
Abstract base class for reflectometry calculators.

This module provides the base class for reflectometry calculators that compute
reflectivity profiles and SLD profiles based on a model. The calculators use
the new corelib CalculatorBase pattern where the calculator is stateful and
holds a reference to the model.
"""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Callable
from typing import Optional

import numpy as np
from easyscience.fitting.calculators.interface_factory import ItemContainer

if TYPE_CHECKING:
    from easyreflectometry.model import Model
    from easyreflectometry.sample import BaseAssembly
    from easyreflectometry.sample import Layer
    from easyreflectometry.sample import Material
    from easyreflectometry.sample import MaterialMixture
    from easyreflectometry.sample import Multilayer

from .wrapper_base import WrapperBase


class CalculatorBase(metaclass=ABCMeta):
    """
    Abstract base class for reflectometry calculators.

    This class provides the common interface and functionality for reflectometry
    calculators. Concrete implementations (Refnx, Refl1d) inherit from this class
    and provide specific wrapper implementations.

    The calculator is stateful and can hold an optional reference to a model.
    This follows the new corelib calculator pattern.

    Class Attributes
    ----------------
    _calculators : list
        Class-level registry of all calculator subclasses for factory discovery.
    name : str
        The name identifier for this calculator.
    """

    _calculators: list[type[CalculatorBase]] = []  # class variable to store all calculators
    name: str = 'base'

    # Property-to-wrapper mapping dictionaries (to be overridden by subclasses)
    _material_link: dict[str, str]
    _layer_link: dict[str, str]
    _item_link: dict[str, str]
    _model_link: dict[str, str]

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs) -> None:
        """Register all non-abstract subclasses for factory discovery.

        :param is_abstract: If True, this subclass won't be added to the registry.
        :param kwargs: Additional keyword arguments passed to parent.
        """
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._calculators.append(cls)

    def __init__(self, model: Optional[Model] = None) -> None:
        """Initialize the calculator.

        :param model: Optional model to associate with this calculator.
                      If provided, bindings will be created automatically.
        """
        self._namespace = {}
        self._wrapper: WrapperBase
        self._model: Optional[Model] = None

        if model is not None:
            self.set_model(model)

    @property
    def model(self) -> Optional[Model]:
        """Get the current model associated with this calculator."""
        return self._model

    def set_model(self, model: Model) -> None:
        """Set the model and create all necessary bindings.

        This method resets the storage and rebuilds all bindings from
        the model's object hierarchy (materials, layers, assemblies, model).

        :param model: The model to associate with this calculator.
        """
        from easyreflectometry.model import Model as ModelClass

        self._model = model
        self.reset_storage()
        self._create_all_bindings(model)

    def _create_all_bindings(self, model: Model) -> None:
        """Create bindings for the entire model hierarchy.

        This walks through the model structure and creates calculator
        bindings for all materials, layers, assemblies, and the model itself.

        :param model: The model to create bindings for.
        """
        from easyreflectometry.model import Model as ModelClass
        from easyreflectometry.sample import BaseAssembly
        from easyreflectometry.sample import Layer
        from easyreflectometry.sample import Material
        from easyreflectometry.sample import MaterialMixture

        # Create materials first
        for assembly in model.sample:
            for layer in assembly.layers:
                material = layer.material
                # Handle both Material and MaterialMixture
                if isinstance(material, (Material, MaterialMixture)):
                    self._create_and_bind(material)

        # Create layers
        for assembly in model.sample:
            for layer in assembly.layers:
                self._create_and_bind(layer)

        # Create assemblies
        for assembly in model.sample:
            self._create_and_bind(assembly)

        # Create the model itself
        self._create_and_bind(model)

    def _create_and_bind(self, obj) -> None:
        """Create calculator objects and bind parameters.

        :param obj: The object (Material, Layer, Assembly, or Model) to create and bind.
        """
        class_links = self.create(obj)
        props = obj._get_linkable_attributes()
        props_names = [prop.name for prop in props]

        for item in class_links:
            for item_key in item.name_conversion.keys():
                if item_key not in props_names:
                    continue
                idx = props_names.index(item_key)
                prop = props[idx]

                # Get value safely
                if hasattr(prop, 'value_no_call_back'):
                    prop_value = prop.value_no_call_back
                else:
                    prop_value = prop.value

                prop._callback = item.make_prop(item_key)
                prop._callback.fset(prop_value)

    def reset_storage(self) -> None:
        """Reset the storage area of the calculator."""
        self._wrapper.reset_storage()

    def create(self, model: Material | Layer | Multilayer | Model) -> list[ItemContainer]:
        """Create calculator objects for the given model component.

        :param model: Object to be created (Material, Layer, Multilayer, or Model).
        :return: List of ItemContainers with binding information.
        """
        from easyreflectometry.model import Model as ModelClass
        from easyreflectometry.sample import BaseAssembly
        from easyreflectometry.sample import Layer
        from easyreflectometry.sample import Material
        from easyreflectometry.sample import MaterialMixture

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
        elif issubclass(t_, ModelClass):
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

        :param material_id: The material name.
        :param layer_id: The layer name.
        """
        self._wrapper.assign_material_to_layer(material_id, layer_id)

    def add_layer_to_item(self, layer_id: str, item_id: str) -> None:
        """Add a layer to the item stack.

        :param layer_id: The layer id.
        :param item_id: The item id.
        """
        self._wrapper.add_layer_to_item(layer_id, item_id)

    def remove_layer_from_item(self, layer_id: str, item_id: str) -> None:
        """Remove a layer from an item stack.

        :param layer_id: The layer id.
        :param item_id: The item id.
        """
        self._wrapper.remove_layer_from_item(layer_id, item_id)

    def add_item_to_model(self, item_id: str, model_id: str) -> None:
        """Add an assembly to the model.

        :param item_id: The assembly/item id.
        :param model_id: The model id.
        """
        self._wrapper.add_item(item_id, model_id)

    def remove_item_from_model(self, item_id: str, model_id: str) -> None:
        """Remove an item from the model.

        :param item_id: The item id.
        :param model_id: The model id.
        """
        self._wrapper.remove_item(item_id, model_id)

    def calculate(self, x_array: np.ndarray) -> np.ndarray:
        """Calculate the reflectivity profile using the current model.

        This is the primary calculation method that uses the bound model.

        :param x_array: Q values to calculate at.
        :return: Reflectivity values at the given Q points.
        :raises ValueError: If no model is set.
        """
        if self._model is None:
            raise ValueError('No model set. Use set_model() first.')
        return self.reflectivity_profile(x_array, self._model.unique_name)

    def reflectivity_profile(self, x_array: np.ndarray, model_id: str) -> np.ndarray:
        """Determine the reflectivity profile for the given range and model.

        :param x_array: Q values to calculate at.
        :param model_id: The model id.
        :return: Reflectivity values.
        """
        return self._wrapper.calculate(x_array, model_id)

    # Keep old name for backwards compatibility
    def reflectity_profile(self, x_array: np.ndarray, model_id: str) -> np.ndarray:
        """Determine the reflectivity profile (legacy name with typo).

        .. deprecated::
            Use reflectivity_profile() instead.

        :param x_array: Q values to calculate at.
        :param model_id: The model id.
        :return: Reflectivity values.
        """
        return self.reflectivity_profile(x_array, model_id)

    def sld_profile(self, model_id: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        :param model_id: The model id. If None, uses the bound model.
        :return: Tuple of (z, sld(z)) arrays.
        :raises ValueError: If no model_id provided and no model is set.
        """
        if model_id is None:
            if self._model is None:
                raise ValueError('No model set. Use set_model() or provide model_id.')
            model_id = self._model.unique_name
        return self._wrapper.sld_profile(model_id)

    def set_resolution_function(self, resolution_function: Callable[[np.ndarray], np.ndarray]) -> None:
        """Set the resolution function for smearing calculations.

        :param resolution_function: The resolution function to use.
        """
        return self._wrapper.set_resolution_function(resolution_function)

    @property
    def include_magnetism(self) -> bool:
        """Get the magnetism flag."""
        return self._wrapper.magnetism

    @include_magnetism.setter
    def include_magnetism(self, magnetism: bool) -> None:
        """Set the magnetism flag for the calculator.

        :param magnetism: True if the calculator should include magnetism.
        """
        self._wrapper.magnetism = magnetism

    @property
    def fit_func(self) -> Callable:
        """Return a fitting function that uses the bound model.

        This provides compatibility with the fitting framework.
        """
        def __fit_func(x_array: np.ndarray, model_id: str) -> np.ndarray:
            return self.reflectivity_profile(x_array, model_id)
        return __fit_func

    def __repr__(self) -> str:
        """Return a string representation of the calculator."""
        model_info = ''
        if self._model is not None:
            model_info = f', model={self._model.unique_name}'
        return f'{self.__class__.__name__}(name={self.name}{model_info})'
