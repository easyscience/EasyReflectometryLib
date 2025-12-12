from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.io.serializer_base import SerializerBase
from easyscience.variable import Parameter
from easyscience.variable.descriptor_base import DescriptorBase

from easyreflectometry.utils import yaml_dump


class BaseCore(ModelBase):
    """Base class for all EasyReflectometry model objects.
    
    This class bridges the new ModelBase API with the legacy 'name' patterns
    used throughout EasyReflectometry. The 'name' property maps to 'display_name' in the
    new architecture.
    
    Note: The 'interface' parameter is deprecated. Calculator binding is now handled
    centrally by the Project class using calculator.set_model().
    """

    def __init__(
        self,
        name: str,
        interface=None,  # Deprecated - kept for backward compatibility
        unique_name: Optional[str] = None,
        **kwargs,
    ):
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)
        super().__init__(unique_name=unique_name, display_name=name)

        # Store kwargs for parameter access (compatibility with ObjBase pattern)
        self._kwargs = kwargs
        for key, value in kwargs.items():
            # Register components with the global object map
            if hasattr(value, 'unique_name'):
                self._global_object.map.add_edge(self, value)
                self._global_object.map.reset_type(value, 'created_internal')

        # Interface is deprecated but kept for backward compatibility
        # It's now a no-op - calculator binding is handled by Project
        self._interface = interface

    def __getattr__(self, name: str):
        """Forward attribute access to _kwargs for ObjBase compatibility."""
        # Check if the name exists in _kwargs (handles both regular and underscore-prefixed kwargs)
        if '_kwargs' in self.__dict__ and name in self._kwargs:
            return self._kwargs[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        """Handle attribute setting for ObjBase compatibility."""
        # During initialization, use normal setting
        if '_kwargs' not in self.__dict__:
            super().__setattr__(name, value)
            return
        # If name is in _kwargs, update the value (for Parameters, set .value)
        if name in self._kwargs:
            existing = self._kwargs[name]
            if isinstance(existing, DescriptorBase) and not isinstance(value, DescriptorBase):
                existing.value = value
            else:
                # Replace the component
                if hasattr(existing, 'unique_name'):
                    self._global_object.map.prune_vertex_from_edge(self, existing)
                self._kwargs[name] = value
                if hasattr(value, 'unique_name'):
                    self._global_object.map.add_edge(self, value)
                    self._global_object.map.reset_type(value, 'created_internal')
        else:
            super().__setattr__(name, value)

    @property
    def name(self) -> str:
        """Get the name of the object (maps to display_name)."""
        return self.display_name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set the name of the object."""
        self.display_name = new_name

    @property
    def interface(self):
        """Get the current interface of the object.
        
        .. deprecated::
            The interface property is deprecated. Calculator binding is now
            handled centrally by the Project class.
        """
        return self._interface

    @interface.setter
    def interface(self, new_interface) -> None:
        """Set the interface (deprecated - now a no-op for sample objects).
        
        .. deprecated::
            The interface property is deprecated. Calculator binding is now
            handled centrally by the Project class using calculator.set_model().
        """
        self._interface = new_interface
        # No longer propagate or generate bindings - this is handled by calculator.set_model()

    def generate_bindings(self) -> None:
        """Generate or re-generate bindings to an interface.
        
        .. deprecated::
            This method is deprecated. Calculator binding is now handled
            centrally by the Project class using calculator.set_model().
        """
        # This is now a no-op for sample objects
        # Calculator binding is handled by calculator.set_model() which calls _create_all_bindings()
        pass

    def _add_component(self, key: str, component) -> None:
        """Dynamically add a component to the class."""
        self._kwargs[key] = component
        self._global_object.map.add_edge(self, component)
        self._global_object.map.reset_type(component, 'created_internal')
        setattr(self, key, component)

    def _get_linkable_attributes(self) -> List[DescriptorBase]:
        """Get all objects which can be linked against as a list.

        :return: List of `Descriptor`/`Parameter` objects.
        """
        item_list = []
        for key, item in self._kwargs.items():
            if hasattr(item, '_get_linkable_attributes'):
                item_list = [*item_list, *item._get_linkable_attributes()]
            elif isinstance(item, DescriptorBase):
                item_list.append(item)
        return item_list

    def get_parameters(self) -> List[Parameter]:
        """Get all parameter objects as a list.

        :return: List of `Parameter` objects.
        """
        par_list = []
        for key, item in self._kwargs.items():
            if hasattr(item, 'get_parameters'):
                par_list = [*par_list, *item.get_parameters()]
            elif isinstance(item, Parameter):
                par_list.append(item)
        return par_list

    def get_fit_parameters(self) -> List[Parameter]:
        """Get all objects which can be fitted (and are not fixed) as a list.

        :return: List of `Parameter` objects which can be used in fitting.
        """
        fit_list = []
        for key, item in self._kwargs.items():
            if hasattr(item, 'get_fit_parameters'):
                fit_list = [*fit_list, *item.get_fit_parameters()]
            elif isinstance(item, Parameter):
                if item.independent and not item.fixed:
                    fit_list.append(item)
        return fit_list

    @abstractmethod
    def _dict_repr(self) -> dict[str, str]: ...

    def __repr__(self) -> str:
        """
        String representation of the layer.

        :return: a string representation of the layer
        :rtype: str
        """
        try:
            return yaml_dump(self._dict_repr)
        except Exception:
            # Fallback for cases where _dict_repr contains non-serializable objects (e.g., mocks)
            return f'{self.__class__.__name__}({self.name})'

    def as_dict(self, skip: Optional[List[str]] = None) -> dict:
        """Produces a cleaned dict using a custom as_dict method.
        The resulting dict matches the parameters in __init__.

        :param skip: List of keys to skip, defaults to `None`.
        :return: Dictionary representation of the object.
        """
        if skip is None:
            skip = []
        
        # Always skip unique_name for nested Parameters to avoid collisions during from_dict
        param_skip = list(skip) + ['unique_name'] if 'unique_name' not in skip else list(skip)
        
        result = {
            '@module': self.__class__.__module__,
            '@class': self.__class__.__name__,
            '@version': None,
        }
        
        # Add name if not default
        if 'name' not in skip:
            result['name'] = self.name
        
        # Add unique_name if not default
        if 'unique_name' not in skip and not self._default_unique_name:
            result['unique_name'] = self.unique_name
        
        # Serialize kwargs - use param_skip for nested objects
        for key, value in self._kwargs.items():
            # Strip leading underscore from key for serialization
            key_name = key.lstrip('_') if key.startswith('_') else key
            if key_name in skip or key in skip:
                continue
            if hasattr(value, 'as_dict'):
                result[key_name] = value.as_dict(skip=param_skip)
            elif hasattr(value, 'to_dict'):
                result[key_name] = value.to_dict(skip=param_skip)
            else:
                result[key_name] = value
        
        return result

    def to_dict(self, skip: Optional[List[str]] = None) -> dict:
        """Convert to dictionary for serialization (alias for as_dict).
        
        This overrides NewBase.to_dict to use our custom serialization.
        
        :param skip: List of keys to skip, defaults to `None`.
        :return: Dictionary representation of the object.
        """
        return self.as_dict(skip=skip)

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> 'BaseCore':
        """
        Re-create an object from a dictionary.

        :param obj_dict: Dictionary containing the serialized contents.
        :return: Reformed object.
        """
        if not SerializerBase._is_serialized_easyscience_object(obj_dict):
            raise ValueError('Input must be a dictionary representing an EasyScience object.')
        
        # Deserialize all values
        kwargs = {}
        for key, value in obj_dict.items():
            if key.startswith('@'):
                continue
            if isinstance(value, dict) and SerializerBase._is_serialized_easyscience_object(value):
                kwargs[key] = SerializerBase._deserialize_value(value)
            else:
                kwargs[key] = value
        
        # Create instance
        return cls(**kwargs)
