# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from typing import Optional

from ..elements.layers.layer import Layer
from .base_collection import BaseCollection


class LayerCollection(BaseCollection):
    def __init__(
        self,
        *layers: Optional[list[Layer]],
        name: str = 'EasyLayerCollection',
        interface=None,
        unique_name: Optional[str] = None,
        populate_if_none: bool = True,  # Needed to match as_dict signature from BaseCollection
        **kwargs,
    ):
        """Init function."""
        if not layers:
            layers = []

        super().__init__(name, interface, unique_name=unique_name, *layers, **kwargs)

    def add_layer(self, layer: Optional[Layer] = None):
        """Add a layer to the collection.

        Parameters
        ----------
        layer : Optional[Layer], optional
            Layer to add. By default, None.
        """
        if layer is None:
            layer = Layer(
                name='EasyLayer added',
                interface=self.interface,
            )
        self.append(layer)

    def duplicate_layer(self, index: int):
        """Duplicate a layer in the collection.

        Parameters
        ----------
        index : int
        layer :
            Assembly to add.
        """
        to_be_duplicated = self[index]
        duplicate = Layer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        duplicate.name = duplicate.name + ' duplicate'
        self.append(duplicate)
