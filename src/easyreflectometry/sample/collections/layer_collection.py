__author__ = 'github.com/arm61'

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
        # Handle layers passed as keyword argument (for compatibility)
        if not layers and 'layers' in kwargs:
            layers = kwargs.pop('layers')
            if not isinstance(layers, (list, tuple)):
                layers = [layers]
        
        if not layers:
            layers = []

        super().__init__(name, interface, unique_name=unique_name, *layers, **kwargs)

    def add_layer(self, layer: Optional[Layer] = None):
        """Add a layer to the collection.

        :param layer: Layer to add.
        """
        if layer is None:
            layer = Layer(
                name='EasyLayer added',
                interface=self.interface,
            )
        self.append(layer)

    def duplicate_layer(self, index: int):
        """Duplicate a layer in the collection.

        :param layer: Assembly to add.
        """
        to_be_duplicated = self[index]
        duplicate = Layer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        duplicate.name = duplicate.name + ' duplicate'
        self.append(duplicate)
