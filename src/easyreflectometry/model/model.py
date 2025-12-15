from __future__ import annotations

__author__ = 'github.com/arm61'

import copy
from numbers import Number
from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter

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
        'max': np.inf,
        'fixed': True,
    },
    'background': {
        'description': 'Linear background to include in reflectometry data',
        'url': 'https://github.com/reflectivity/edu_outreach/blob/master/refl_maths/paper.tex',
        'value': 1e-8,
        'min': 0.0,
        'max': np.inf,
        'fixed': True,
    },
    'resolution': {
        'value': 5.0,
    },
}

COLORS = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9']


class Model(ModelBase):
    """Model is the class that represents the experiment.
    It is used to store the information about the experiment and to perform the calculations.
    """

    # Class attributes for type hints
    _sample: Sample
    _scale: Parameter
    _background: Parameter

    def __init__(
        self,
        sample: Union[Sample, None] = None,
        scale: Union[Parameter, Number, None] = None,
        background: Union[Parameter, Number, None] = None,
        resolution_function: Union[ResolutionFunction, None] = None,
        name: str = 'Model',
        color: str = COLORS[0],
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        :param sample: The sample being modelled.
        :param scale: Scaling factor of profile.
        :param background: Linear background magnitude.
        :param name: Name of the model, defaults to 'Model'.
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
            unique_name=unique_name,
            display_name=name,
        )

        # Store components and register with global object map
        self._sample = sample
        self._scale = scale
        self._background = background
        self._global_object.map.add_edge(self, sample)
        self._global_object.map.add_edge(self, scale)
        self._global_object.map.add_edge(self, background)
        self._global_object.map.reset_type(sample, 'created_internal')
        self._global_object.map.reset_type(scale, 'created_internal')
        self._global_object.map.reset_type(background, 'created_internal')

        self._resolution_function = resolution_function

        # Interface handling - for Model, we DO trigger binding generation when interface is set
        # because Model is the top-level object that contains all sample information.
        # This provides backward compatibility with code that uses Model(interface=factory).
        # Sample objects no longer trigger bindings when interface is set (they're just data).
        self._interface = None
        if interface is not None:
            self.interface = interface  # This calls the setter which triggers generate_bindings

    @property
    def name(self) -> str:
        """Get the name of the model (maps to display_name)."""
        return self.display_name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set the name of the model."""
        self.display_name = new_name

    @property
    def sample(self) -> Sample:
        """Get the sample."""
        return self._sample

    @sample.setter
    def sample(self, new_sample: Sample) -> None:
        """Set the sample."""
        old_sample = self._sample
        self._sample = new_sample
        self._global_object.map.prune_vertex_from_edge(self, old_sample)
        self._global_object.map.add_edge(self, new_sample)
        self._global_object.map.reset_type(new_sample, 'created_internal')

    @property
    def scale(self) -> Parameter:
        """Get the scale parameter."""
        return self._scale

    @scale.setter
    def scale(self, value: Union[Parameter, Number]) -> None:
        """Set the scale value."""
        if isinstance(value, Parameter):
            old_scale = self._scale
            self._scale = value
            self._global_object.map.prune_vertex_from_edge(self, old_scale)
            self._global_object.map.add_edge(self, value)
            self._global_object.map.reset_type(value, 'created_internal')
        else:
            self._scale.value = value

    @property
    def background(self) -> Parameter:
        """Get the background parameter."""
        return self._background

    @background.setter
    def background(self, value: Union[Parameter, Number]) -> None:
        """Set the background value."""
        if isinstance(value, Parameter):
            old_background = self._background
            self._background = value
            self._global_object.map.prune_vertex_from_edge(self, old_background)
            self._global_object.map.add_edge(self, value)
            self._global_object.map.reset_type(value, 'created_internal')
        else:
            self._background.value = value

    def add_assemblies(self, *assemblies: list[BaseAssembly]) -> None:
        """Add assemblies to the model sample.

        :param assemblies: Assemblies to add to model sample.
        """
        if not assemblies:
            self.sample.add_assembly()
        else:
            for assembly in assemblies:
                if issubclass(assembly.__class__, BaseAssembly):
                    self.sample.add_assembly(assembly)
                else:
                    raise ValueError(f'Object {assembly} is not a valid type, must be a child of BaseAssembly.')
        # Regenerate all bindings after adding assemblies
        if self.interface is not None:
            self.generate_bindings()

    def duplicate_assembly(self, index: int) -> None:
        """Duplicate a given item or layer in a sample.

        :param idx: Index of the item or layer to duplicate
        """
        self.sample.duplicate_assembly(index)
        # Regenerate all bindings after duplicating assembly
        if self.interface is not None:
            self.generate_bindings()

    def remove_assembly(self, index: int) -> None:
        """Remove an assembly from the model.

        :param idx: Index of the item to remove.
        """
        self.sample.remove_assembly(index)
        # Regenerate all bindings after removing assembly
        if self.interface is not None:
            self.generate_bindings()

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
        """Get the current interface of the object.
        
        .. deprecated::
            The interface property is deprecated. Calculator binding is now
            handled centrally by the Project class using calculator.set_model().
        """
        return self._interface

    @interface.setter
    def interface(self, new_interface) -> None:
        """Set the interface for the model.
        
        .. deprecated::
            The interface property is deprecated. Calculator binding is now
            handled centrally by the Project class using calculator.set_model().
        """
        self._interface = new_interface
        # For backward compatibility, if interface has generate_bindings, call it
        if new_interface is not None and hasattr(new_interface, 'generate_bindings'):
            new_interface.generate_bindings(self)
            if hasattr(new_interface, '__call__') and hasattr(new_interface(), 'set_resolution_function'):
                new_interface().set_resolution_function(self._resolution_function)

    def generate_bindings(self) -> None:
        """Generate or re-generate bindings to an interface.
        
        .. deprecated::
            This method is deprecated. Calculator binding is now handled
            centrally by the Project class using calculator.set_model().
        """
        if self.interface is None:
            return  # No-op if no interface
        # For backward compatibility
        if hasattr(self.interface, 'generate_bindings'):
            self.interface.generate_bindings(self)

    def _get_linkable_attributes(self) -> list:
        """Get all objects which can be linked against as a list.

        :return: List of `Descriptor`/`Parameter` objects.
        """
        from easyscience.variable.descriptor_base import DescriptorBase

        item_list = []
        for attr in [self._scale, self._background, self._sample]:
            if hasattr(attr, '_get_linkable_attributes'):
                item_list.extend(attr._get_linkable_attributes())
            elif isinstance(attr, DescriptorBase):
                item_list.append(attr)
        return item_list

    def get_parameters(self) -> list:
        """Get all parameter objects as a list.

        :return: List of `Parameter` objects.
        """
        from easyscience.variable import Parameter

        par_list = []
        for attr in [self._scale, self._background, self._sample]:
            if hasattr(attr, 'get_parameters'):
                par_list.extend(attr.get_parameters())
            elif isinstance(attr, Parameter):
                par_list.append(attr)
        return par_list

    def get_fit_parameters(self) -> list:
        """Get all objects which can be fitted (and are not fixed) as a list.

        :return: List of `Parameter` objects which can be used in fitting.
        """
        from easyscience.variable import Parameter

        fit_list = []
        for attr in [self._scale, self._background, self._sample]:
            if hasattr(attr, 'get_fit_parameters'):
                fit_list.extend(attr.get_fit_parameters())
            elif isinstance(attr, Parameter):
                if attr.independent and not attr.fixed:
                    fit_list.append(attr)
        return fit_list

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
        
        # Always skip unique_name for nested Parameters to avoid collisions during from_dict
        param_skip = list(skip) + ['unique_name'] if 'unique_name' not in skip else list(skip)
        
        this_dict = {
            '@module': self.__class__.__module__,
            '@class': self.__class__.__name__,
            '@version': None,
            'name': self.name,
            'color': self.color,
        }
        
        # Add unique_name if not default
        if 'unique_name' not in skip and not self._default_unique_name:
            this_dict['unique_name'] = self.unique_name
        
        # Add sample - use param_skip to avoid parameter unique_name collisions
        this_dict['sample'] = self.sample.as_dict(skip=param_skip)
        
        # Add scale and background - use param_skip
        if 'scale' not in skip:
            this_dict['scale'] = self._scale.as_dict(skip=param_skip)
        if 'background' not in skip:
            this_dict['background'] = self._background.as_dict(skip=param_skip)
        
        # Add resolution function
        this_dict['resolution_function'] = self.resolution_function.as_dict(skip=param_skip)
        
        # Add interface
        if self.interface is None:
            this_dict['interface'] = None
        else:
            this_dict['interface'] = self.interface().name
        
        return this_dict

    def to_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Convert to dictionary for serialization (alias for as_dict).
        
        This overrides NewBase.to_dict to use our custom serialization.
        
        :param skip: List of keys to skip, defaults to `None`.
        :return: Dictionary representation of the model.
        """
        return self.as_dict(skip=skip)

    def as_orso(self) -> dict:
        """Convert the model to a dictionary suitable for ORSO."""
        this_dict = self.as_dict()

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
