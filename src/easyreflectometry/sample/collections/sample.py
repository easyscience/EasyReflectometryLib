from __future__ import annotations

__author__ = 'github.com/arm61'

from typing import List
from typing import Optional

from ..assemblies.base_assembly import BaseAssembly
from ..assemblies.multilayer import Multilayer
from ..assemblies.repeating_multilayer import RepeatingMultilayer
from ..assemblies.surfactant_layer import SurfactantLayer
from ..elements.layers.layer import Layer
from .base_collection import BaseCollection


# Needs to be a function, elements are added to the global_object.map
def DEFAULT_ELEMENTS(interface):
    """:meta private:"""
    return (
        Multilayer(interface=interface),
        Multilayer(interface=interface),
    )


class Sample(BaseCollection):
    """A sample is a collection of assemblies that represent the structure for which experimental measurements exist."""

    def __init__(
        self,
        *assemblies: Optional[List[BaseAssembly]],
        name: str = 'EasySample',
        interface=None,
        unique_name: Optional[str] = None,
        populate_if_none: bool = True,
        **kwargs,
    ):
        """Constructor.

        :param args: The assemblies in the sample.
        :param name: Name of the sample, defaults to 'EasySample'.
        :param interface: Calculator interface, defaults to `None`.
        """
        if not assemblies:
            if populate_if_none:
                assemblies = DEFAULT_ELEMENTS(interface)
            else:
                assemblies = []

        for assembly in assemblies:
            if not issubclass(type(assembly), BaseAssembly):
                raise ValueError('The elements must be an Assembly.')
        super().__init__(name, interface, unique_name=unique_name, *assemblies, **kwargs)

    def add_assembly(self, assembly: Optional[BaseAssembly] = None):
        """Add an assembly to the sample.

        :param assembly: Assembly to add.
        """
        if assembly is None:
            assembly = Multilayer(
                name='EasyMultilayer added',
                interface=self.interface,
            )
        self.append(assembly)

    def duplicate_assembly(self, index: int):
        """Add an assembly to the sample.

        :param assembly: Assembly to add.
        """
        to_be_duplicated = self[index]
        if isinstance(to_be_duplicated, Multilayer):
            duplicate = Multilayer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        elif isinstance(to_be_duplicated, RepeatingMultilayer):
            duplicate = RepeatingMultilayer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        elif isinstance(to_be_duplicated, SurfactantLayer):
            duplicate = SurfactantLayer.from_dict(to_be_duplicated.as_dict(skip=['unique_name']))
        duplicate.name = duplicate.name + ' duplicate'
        self.append(duplicate)

    def move_up(self, index: int):
        """Move the assembly at the given index up in the sample.

        :param index: Index of the assembly to move up.
        """
        super().move_up(index)

    def move_down(self, index: int):
        """Move the assembly at the given index down in the sample.

        :param index: Index of the assembly to move down.
        """
        super().move_down(index)

    def remove_assembly(self, index: int):
        """Remove the assembly at the given index from the sample.

        :param index: Index of the assembly to remove.
        """
        self.pop(index)

    @property
    def superphase(self) -> Layer:
        """The superphase of the sample."""
        return self[0].front_layer

    @property
    def subphase(self) -> Layer:
        """The subphase of the sample."""
        # This assembly only got one layer
        if self[-1].back_layer is None:
            return self[-1].front_layer
        else:
            return self[-1].back_layer

    # Representation
    def as_dict(self, skip: Optional[List[str]] = None) -> dict:
        """Produces a cleaned dict using a custom as_dict method to skip necessary things.
        The resulting dict matches the parameters in __init__

        :param skip: List of keys to skip, defaults to `None`.
        """
        this_dict = super().as_dict(skip=skip)
        this_dict['populate_if_none'] = self.populate_if_none
        return this_dict
