__author__ = 'github.com/arm61'

from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.Objects.new_variable import Parameter

from easyreflectometry.utils import get_as_parameter

from ...base_core import BaseCore

DEFAULTS = {
    'sld': {
        'description': 'The real scattering length density for a material in e-6 per squared angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 4.186,
        'unit': '1 / angstrom^2',
        'min': -np.Inf,
        'max': np.Inf,
        'fixed': True,
    },
    'isld': {
        'description': 'The imaginary scattering length density for a material in e-6 per squared angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 0.0,
        'unit': '1 / angstrom^2',
        'min': -np.Inf,
        'max': np.Inf,
        'fixed': True,
    },
}


class Material(BaseCore):
    # Added in super().__init__
    sld: Parameter
    isld: Parameter

    def __init__(
        self,
        sld: Union[Parameter, float, None] = None,
        isld: Union[Parameter, float, None] = None,
        name: str = 'EasyMaterial',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        :param sld: Real scattering length density.
        :param isld: Imaginary scattering length density.
        :param name: Name of the material, defaults to 'EasyMaterial'.
        :param interface: Calculator interface, defaults to `None`.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        sld = get_as_parameter(
            name='sld',
            value=sld,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Sld',
        )
        isld = get_as_parameter(
            name='isld',
            value=isld,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Isld',
        )

        super().__init__(
            name=name,
            sld=sld,
            isld=isld,
            interface=interface,
            unique_name=unique_name,
        )

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            self.name: {
                'sld': f'{self.sld.value:.3f}e-6 {self.sld.unit}',
                'isld': f'{self.isld.value:.3f}e-6 {self.isld.unit}',
            }
        }
