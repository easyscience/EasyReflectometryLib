__author__ = 'github.com/arm61'

from easyscience.fitting.calculators.interface_factory import ItemContainer

from easyreflectometry.model import Model
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer

from ..calculator_base import CalculatorBase
from .wrapper import BornAgainWrapper

"""
THIS CODE IS NOT FUNCTIONAL
PLEASE CONSULT ONE OF THE OTHER CALCULATORS FOR A FUNCTIONAL EXAMPLE
"""


class BornAgain(CalculatorBase):
    """
    Calculator for BornAgain
    """

    name = 'BornAgain'

    _material_link = {
        'sld': 'real',
        'isld': 'imag',
    }

    _layer_link = {
        'thickness': 'thickness',
        'roughness': 'sigma',
    }

    _item_link = {
        'repetitions': 'repeats',
    }

    _model_link = {
        'scale': 'scale',
        'background': 'background',
        'resolution': 'resolution',
    }

    def __init__(self):
        super().__init__()
        self._wrapper = BornAgainWrapper()

    def reset_storage(self) -> None:
        """
        Reset the storage area of the calculator
        """
        self._wrapper.reset_storage()

    def create(self, model: Material | Layer | Multilayer | Model) -> list[ItemContainer]:
        """
        Creation function

        :param model: Object to be created
        :type model: Union[Material, Layer, Item, Model]
        :return: Item containers of the objects
        :rtype: List[ItemContainer]
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
        elif issubclass(t_, Multilayer):
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
            self._wrapper.create_model()
            r_list.append(
                ItemContainer(
                    'model',
                    self._model_link,
                    self._wrapper.get_model_value,
                    self._wrapper.update_model,
                )
            )
            for i in model.structure:
                self.add_item_to_model(i.unique_name)
        return r_list

    def assign_material_to_layer(self, material_id: int, layer_id: int) -> None:
        """
        Assign a material to a layer.

        :param material_name: The material name
        :type material_name: str
        :param layer_name: The layer name
        :type layer_name: str
        """
        self._wrapper.assign_material_to_layer(material_id, layer_id)

    def add_layer_to_item(self, layer_id: int, item_id: int) -> None:
        """
        Add a layer to the item stack

        :param item_id: The item id
        :type item_id: int
        :param layer_id: The layer id
        :type layer_id: int
        """
        self._wrapper.add_layer_to_item(layer_id, item_id)

    def remove_layer_from_item(self, layer_id: int, item_id: int) -> None:
        """
        Remove a layer from an item stack

        :param item_id: The item id
        :param layer_id: The layer id
        """
        self._wrapper.remove_layer_from_item(layer_id, item_id)

    def add_item_to_model(self, item_id: int) -> None:
        """
        Add a layer to the item stack

        :param item_id: The item id
        :type item_id: int
        """
        self._wrapper.add_item(item_id)

    def remove_item_from_model(self, item_id: int) -> None:
        """
        Remove a layer from the item stack

        :param item_id: The item id
        :type item_id: int
        :param layer_id: The layer id
        :type layer_id: int
        """
        self._wrapper.remove_item(item_id)
