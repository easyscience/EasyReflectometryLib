from copy import deepcopy
from numbers import Number
from typing import Optional
from typing import Union

import yaml
from easyscience import global_object
from easyscience.Objects.new_variable import Parameter


def get_as_parameter(
    name: str,
    value: Union[Parameter, Number, None],
    default_dict: dict,
    unique_name_prefix: Optional[str] = None,
) -> Parameter:
    """
    This function creates a parameter for the variable `name`.  A parameter has a value and metadata.
    If the value already is a parameter, it is returned.
    If the value is a number, a parameter is created with this value and metadata from the dictionary.
    If the value is None, a parameter is created with the default value and metadata from the dictionary.

    param value: The value to use for the parameter.  If None, the default value in the dictionary is used.
    param name: The name of the parameter
    param default_dict: Dictionary with entry for `name` containing the default value and metadata for the parameter
    """
    # This is a parameter, return it
    if isinstance(value, Parameter):
        return value

    # Ensure we got the dictionary for the parameter with the given name
    # Should leave the passed dictionary unchanged
    if name not in default_dict:
        parameter_dict = deepcopy(default_dict)
    else:
        parameter_dict = deepcopy(default_dict[name])

    # Add specific unique name prefix if requested
    if unique_name_prefix is not None:
        parameter_dict['unique_name'] = global_object.generate_unique_name(unique_name_prefix + 'Parameter')

    if value is None:
        # Create a default parameter using both value and metadata from dictionary
        return Parameter(name, **parameter_dict)
    elif isinstance(value, Number):
        # Create a parameter using provided value and metadata from dictionary
        del parameter_dict['value']
        return Parameter(name, value, **parameter_dict)

    raise ValueError(f'{name} must be a Parameter, a number, or None.')


def yaml_dump(dict_repr: dict) -> str:
    return yaml.dump(dict_repr, sort_keys=False, allow_unicode=True)


def collect_unique_names_from_dict(structure_dict: dict, unique_names: Optional[list[str]] = None) -> dict:
    """
    This function returns a list with the 'unique_name' found the input dictionary.
    """
    if unique_names is None:
        unique_names = []

    if isinstance(structure_dict, dict):
        for key, value in structure_dict.items():
            if isinstance(value, dict):
                collect_unique_names_from_dict(value, unique_names)
            elif isinstance(value, list):
                for element in value:
                    collect_unique_names_from_dict(element, unique_names)
            if key == 'unique_name':
                unique_names.append(value)
    return unique_names


def count_free_parameters(project) -> int:
    return sum(1 for parameter in project.parameters if parameter.free)


def count_fixed_parameters(project) -> int:
    return sum(1 for parameter in project.parameters if not parameter.free)


def count_parameter_user_constraints(project) -> int:
    return sum(len(parameter.user_constraints.keys()) for parameter in project.parameters if not parameter.free)
