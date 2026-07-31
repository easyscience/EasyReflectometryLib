# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.limits import apply_default_limits
from easyreflectometry.utils import get_as_parameter

from ...base_core import BaseCore

DEFAULTS = {
    'sld': {
        'description': 'The real scattering length density for a material in e-6 per squared angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 4.186,
        'unit': '1 / angstrom^2',
        'min': -np.inf,
        'max': np.inf,
        'fixed': True,
    },
    'isld': {
        'description': 'The imaginary scattering length density for a material in e-6 per squared angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 0.0,
        'unit': '1 / angstrom^2',
        'min': -np.inf,
        'max': np.inf,
        'fixed': True,
    },
}


class Material(BaseCore):
    def __init__(
        self,
        sld: Union[Parameter, float, None] = None,
        isld: Union[Parameter, float, None] = None,
        name: str = 'EasyMaterial',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        sld : Union[Parameter, float, None], optional
            Real scattering length density. By default, None.
        isld : Union[Parameter, float, None], optional
            Imaginary scattering length density. By default, None.
        name : str, optional
            Name of the material. By default, 'EasyMaterial'.
        interface :
            Calculator interface. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        sld = get_as_parameter(
            name='sld',
            value=sld,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Sld',
        )
        apply_default_limits(sld, 'sld')

        isld = get_as_parameter(
            name='isld',
            value=isld,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Isld',
        )
        apply_default_limits(isld, 'isld')

        super().__init__(name=name, unique_name=unique_name)
        self._sld = sld
        self._isld = isld

        if interface is not None:
            self.interface = interface

    @property
    def sld(self) -> Parameter:
        return self._sld

    @sld.setter
    def sld(self, value: float) -> None:
        self._sld.value = value

    @property
    def isld(self) -> Parameter:
        return self._isld

    @isld.setter
    def isld(self, value: float) -> None:
        self._isld.value = value

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            self.name: {
                'sld': f'{self._sld.value:.3f}e-6 {self._sld.unit}',
                'isld': f'{self._isld.value:.3f}e-6 {self._isld.unit}',
            }
        }
