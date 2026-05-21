# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod

import numpy as np

from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import ResolutionFunction


class WrapperBase:
    def __init__(self):
        """Constructor."""
        self._magnetism = False
        self.storage = {
            'material': {},
            'layer': {},
            'item': {},
            'model': {},
        }
        self._resolution_function = PercentageFwhm()

    def reset_storage(self):
        """Reset the storage area to blank."""
        self.storage = {
            'material': {},
            'layer': {},
            'item': {},
            'model': {},
        }

    @abstractmethod
    def create_material(self, name: str):
        """Create a material using SLD.

        Parameters
        ----------
        name : str
            The name of the material.
        """
        ...

    @abstractmethod
    def create_layer(self, name: str):
        """Create a layer using Slab.

        Parameters
        ----------
        name : str
            The name of the layer.
        """
        ...

    @abstractmethod
    def create_item(self, name: str):
        """Create an item using Stack.

        Parameters
        ----------
        name : str
            The name of the item.
        """
        ...

    @abstractmethod
    def create_model(self, name: str):
        """Create a model for analysis.

        Parameters
        ----------
        name : str
            Name for the model.
        """
        ...

    @abstractmethod
    def update_model(self, name: str, **kwargs):
        """Update the non-structural parameters of the model.

        Parameters
        ----------
        name : str
            Name for the model.
        **kwargs :
        """
        ...

    @abstractmethod
    def get_model_value(self, name: str, key: str) -> float:
        """A function to get a given model value.

        Parameters
        ----------
        name : str
            Name for the model.
        key : str
            The given value keys.
        """
        ...

    @abstractmethod
    def assign_material_to_layer(self, material_name: str, layer_name: str):
        """Assign a material to a layer.

        Parameters
        ----------
        material_name : str
            The material name.
        layer_name : str
            The layer name.
        """
        ...

    @abstractmethod
    def add_layer_to_item(self, layer_name: str, item_name: str):
        """Create a layer from the material of the same name, in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        ...

    @abstractmethod
    def add_item(self, item_name: str, model_name: str):
        """Add an item to the model.

        Parameters
        ----------
        item_name : str
            Items to add to model.
        model_name : str
            Name for the model.
        """
        ...

    @abstractmethod
    def remove_layer_from_item(self, layer_name: str, item_name: str):
        """Remove a layer in a given item.

        Parameters
        ----------
        layer_name : str
            The layer name.
        item_name : str
            The item name.
        """
        ...

    @abstractmethod
    def remove_item(self, item_name: str, model_name: str):
        """Remove a given item.

        Parameters
        ----------
        item_name : str
            The item name.
        model_name : str
            Name of the model.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def sld_profile(self, model_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_name : str
            Name for the model.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Z and sld(z).
        """
        ...

    def update_material(self, name: str, **kwargs):
        """Update a material.

        Parameters
        ----------
        name : str
            The name of the material.
        **kwargs :
            Key-value pairs of attributes to update.
        """
        material = self.storage['material'][name]
        for key in kwargs.keys():
            item = getattr(material, key)
            setattr(item, 'value', kwargs[key])

    def get_material_value(self, name: str, key: str) -> float:
        """A function to get a given material value.

        Parameters
        ----------
        name : str
            The material name.
        key : str
            The given value keys.

        Returns
        -------
        float
            The desired value.
        """
        material = self.storage['material'][name]
        item = getattr(material, key)
        return getattr(item, 'value')

    def update_layer(self, name: str, **kwargs):
        """Update a layer in a given item.

        Parameters
        ----------
        name : str
            The layer name.
        **kwargs :
        """
        layer = self.storage['layer'][name]
        for key in kwargs.keys():
            ii = getattr(layer, key)
            setattr(ii, 'value', kwargs[key])

    def get_layer_value(self, name: str, key: str) -> float:
        """A function to get a given layer value.

        Parameters
        ----------
        name : str
            The layer name.
        key : str
            The given value keys.
        """
        layer = self.storage['layer'][name]
        ii = getattr(layer, key)
        return getattr(ii, 'value')

    def update_item(self, name: str, **kwargs):
        """Update a layer.

        Parameters
        ----------
        **kwargs :
        name : str
            The item name.
        """
        item = self.storage['item'][name]
        for key in kwargs.keys():
            ii = getattr(item, key)
            setattr(ii, 'value', kwargs[key])

    def get_item_value(self, name: str, key: str) -> float:
        """A function to get a given item value.

        Parameters
        ----------
        name : str
            The item name.
        key : str
            The given value keys.

        Returns
        -------
        float
            The desired value.
        """
        item = self.storage['item'][name]
        item = getattr(item, key)
        return getattr(item, 'value')

    def __getstate__(self) -> dict:
        return {
            'storage': self.storage,
            'resolution_function': self._resolution_function,
            'magnetism': self._magnetism,
        }

    def __setstate__(self, state: dict) -> None:
        self.storage = state['storage']
        self._resolution_function = state['resolution_function']
        self._magnetism = state['magnetism']

    def set_resolution_function(self, resolution_function: ResolutionFunction) -> None:
        """Set the resolution function for the calculator.

        Parameters
        ----------
        resolution_function : ResolutionFunction
            The resolution function.
        """
        self._resolution_function = resolution_function

    @property
    def magnetism(self) -> bool:
        """Magnetism function."""
        return self._magnetism

    @magnetism.setter
    def magnetism(self, magnetism: bool) -> None:
        """Set the magnetism flag.

        Parameters
        ----------
        magnetism : bool
            The magnetism flag.
        """
        self._magnetism = magnetism
