from typing import Optional
from typing import Union

from easyscience import global_object
from easyscience.Objects.new_variable import Parameter

from easyreflectometry.utils import get_as_parameter

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer import Layer
from .multilayer import Multilayer

DEFAULTS = {
    'repetitions': {
        'description': 'Number of repetitions of the given series of layers',
        'value': 1,
        'min': 1,
        'max': 9999,
        'fixed': True,
    }
}


class RepeatingMultilayer(Multilayer):
    """
    A repeating multi layer is build from a `Multilayer` and which it repeats
    for a given number of times. This enables a computational efficiency in many
    reflectometry engines as the operation can be performed for a single
    `Multilayer` and cheaply combined for the appropriate number of
    `repetitions`.

    More information about the usage of this assembly is available in the
    `repeating multilayer documentation`_

    .. _`repeating multilayer documentation`: ../sample/assemblies_library.html#repeatingmultilayer
    """

    def __init__(
        self,
        layers: Union[LayerCollection, Layer, list[Layer], None] = None,
        repetitions: Union[Parameter, int, None] = None,
        name: str = 'EasyRepeatingMultilayer',
        unique_name: Optional[str] = None,
        interface=None,
        populate_if_none: bool = True,
    ):
        """Constructor.

        :param layers: The layers that make up the multi-layer that will be repeated.
        :param repetitions: Number of repetitions of the given series of layers
        :param name: Name for the repeating multi layer, defaults to 'EasyRepeatingMultilayer'.
        :param interface: Calculator interface, defaults to `None`.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if layers is None:
            if populate_if_none:
                layers = LayerCollection([Layer(interface=interface)])
            else:
                layers = LayerCollection()
        elif isinstance(layers, Layer):
            layers = LayerCollection(layers, name=layers.name)
        elif isinstance(layers, list):
            layers = LayerCollection(*layers, name='/'.join([layer.name for layer in layers]))
        # Needed to ensure an empty list is created when saving and instatiating the object as_dict -> from_dict
        # Else collisions might occur in global_object.map
        self.populate_if_none = False

        repetitions = get_as_parameter(
            name='repetitions',
            value=repetitions,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Repetitions',
        )

        super().__init__(
            layers=layers,
            name=name,
            interface=interface,
            type='Repeating Multi-layer',
        )
        self._add_component('repetitions', repetitions)
        self.interface = interface

    # Representation
    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        d_dict = {self.name: self.layers._dict_repr}
        d_dict[self.name]['repetitions'] = float(self.repetitions.value)
        return d_dict
