from __future__ import annotations

__author__ = 'github.com/arm61'

import copy
from numbers import Number
from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.Objects.new_variable import Parameter
from easyscience.Objects.ObjectClasses import BaseObj

from easyreflectometry.sample import BaseAssembly
from easyreflectometry.sample import Sample
from easyreflectometry.utils import get_as_parameter
from easyreflectometry.utils import yaml_dump

from .resolution_functions import PercentageFwhm
from .resolution_functions import ResolutionFunction

DEFAULTS = {
    'scale': {
        'description': 'Scaling of the reflectomety profile',
        'url': 'https://github.com/reflectivity/edu_outreach/blob/master/refl_maths/paper.tex',
        'value': 1.0,
        'min': 0,
        'max': np.Inf,
        'fixed': True,
    },
    'background': {
        'description': 'Linear background to include in reflectometry data',
        'url': 'https://github.com/reflectivity/edu_outreach/blob/master/refl_maths/paper.tex',
        'value': 1e-8,
        'min': 0.0,
        'max': np.Inf,
        'fixed': True,
    },
    'resolution': {
        'value': 5.0,
    },
}


class Model(BaseObj):
    """Model is the class that represents the experiment.
    It is used to store the information about the experiment and to perform the calculations.
    """

    # Added in super().__init__
    name: str
    sample: Sample
    scale: Parameter
    background: Parameter

    def __init__(
        self,
        sample: Union[Sample, None] = None,
        scale: Union[Parameter, Number, None] = None,
        background: Union[Parameter, Number, None] = None,
        resolution_function: Union[ResolutionFunction, None] = None,
        name: str = 'EasyModel',
        color: str = 'black',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        :param sample: The sample being modelled.
        :param scale: Scaling factor of profile.
        :param background: Linear background magnitude.
        :param name: Name of the model, defaults to 'EasyModel'.
        :param resolution_function: Resolution function, defaults to PercentageFwhm.
        :param interface: Calculator interface, defaults to `None`.

        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if sample is None:
            sample = Sample(interface=interface)
        if resolution_function is None:
            resolution_function = PercentageFwhm(DEFAULTS['resolution']['value'])

        scale = get_as_parameter('scale', scale, DEFAULTS)
        background = get_as_parameter('background', background, DEFAULTS)
        self.color = color

        super().__init__(
            name=name,
            unique_name=unique_name,
            sample=sample,
            scale=scale,
            background=background,
        )
        self.resolution_function = resolution_function

        # Must be set after resolution function
        self.interface = interface

    def add_assemblies(self, *assemblies: list[BaseAssembly]) -> None:
        """Add assemblies to the model sample.

        :param assemblies: Assemblies to add to model sample.
        """
        if not assemblies:
            self.sample.add_assembly()
            if self.interface is not None:
                self.interface().add_item_to_model(self.sample[-1].unique_name, self.unique_name)
        else:
            for assembly in assemblies:
                if issubclass(assembly.__class__, BaseAssembly):
                    self.sample.add_assembly(assembly)
                    if self.interface is not None:
                        self.interface().add_item_to_model(self.sample[-1].unique_name, self.unique_name)
                else:
                    raise ValueError(f'Object {assembly} is not a valid type, must be a child of BaseAssembly.')

    def duplicate_assembly(self, index: int) -> None:
        """Duplicate a given item or layer in a sample.

        :param idx: Index of the item or layer to duplicate
        """
        self.sample.duplicate_assembly(index)
        if self.interface is not None:
            self.interface().add_item_to_model(self.sample[-1].unique_name, self.unique_name)

    def remove_assembly(self, index: int) -> None:
        """Remove an assembly from the model.

        :param idx: Index of the item to remove.
        """
        assembly_unique_name = self.sample[index].unique_name
        self.sample.remove_assembly(index)
        if self.interface is not None:
            self.interface().remove_item_from_model(assembly_unique_name, self.unique_name)

    @property
    def resolution_function(self) -> ResolutionFunction:
        """Return the resolution function."""
        return self._resolution_function

    @resolution_function.setter
    def resolution_function(self, resolution_function: ResolutionFunction) -> None:
        """Set the resolution function for the model."""
        self._resolution_function = resolution_function
        if self.interface is not None:
            self.interface().set_resolution_function(self._resolution_function)

    @property
    def interface(self):
        """
        Get the current interface of the object
        """
        return self._interface

    @interface.setter
    def interface(self, new_interface) -> None:
        """Set the interface for the model."""
        # From super class
        self._interface = new_interface
        if new_interface is not None:
            self.generate_bindings()
            self._interface().set_resolution_function(self._resolution_function)

    # Representation
    @property
    def _dict_repr(self) -> dict[str, dict[str, str]]:
        """A simplified dict representation."""
        if isinstance(self._resolution_function, PercentageFwhm):
            resolution_value = self._resolution_function.as_dict()['constant']
            resolution = f'{resolution_value} %'
        else:
            resolution = 'function of Q'

        return {
            self.name: {
                'scale': float(self.scale.value),
                'background': float(self.background.value),
                'resolution': resolution,
                'color': self.color,
                'sample': self.sample._dict_repr,
            }
        }

    def __repr__(self) -> str:
        """String representation of the layer."""
        return yaml_dump(self._dict_repr)

    def as_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Produces a cleaned dict using a custom as_dict method to skip necessary things.
        The resulting dict matches the parameters in __init__

        :param skip: List of keys to skip, defaults to `None`.
        """
        if skip is None:
            skip = []
        skip.extend(['sample', 'resolution_function', 'interface'])
        this_dict = super().as_dict(skip=skip)
        this_dict['sample'] = self.sample.as_dict(skip=skip)
        this_dict['resolution_function'] = self.resolution_function.as_dict(skip=skip)
        if self.interface is None:
            this_dict['interface'] = None
        else:
            this_dict['interface'] = self.interface().name
        return this_dict

    @classmethod
    def from_dict(cls, passed_dict: dict) -> Model:
        """
        Create a Model from a dictionary.

        :param this_dict: dictionary of the Model
        :return: Model
        """
        # Causes circular import if imported at the top
        from easyreflectometry.calculators import CalculatorFactory

        this_dict = copy.deepcopy(passed_dict)
        resolution_function = ResolutionFunction.from_dict(this_dict['resolution_function'])
        del this_dict['resolution_function']
        interface_name = this_dict['interface']
        del this_dict['interface']
        if interface_name is not None:
            interface = CalculatorFactory()
            interface.switch(interface_name)
        else:
            interface = None

        model = super().from_dict(this_dict)

        model.resolution_function = resolution_function
        model.interface = interface
        return model
