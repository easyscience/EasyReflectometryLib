# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.special.calculations import density_to_sld
from easyreflectometry.special.calculations import molecular_weight as compute_molecular_weight
from easyreflectometry.special.calculations import neutron_scattering_length
from easyreflectometry.utils import get_as_parameter

from .material import DEFAULTS as MATERIAL_DEFAULTS
from .material import Material

DEFAULTS = {
    'chemical_structure': 'Si',
    'density': {
        'description': 'The mass density of the material.',
        'url': 'https://en.wikipedia.org/wiki/Density',
        'value': 2.33,
        'unit': 'gram / centimeter ** 3',
        'min': 0,
        'max': np.inf,
        'fixed': True,
    },
    'molecular_weight': {
        'description': 'The molecular weight of a material.',
        'url': 'https://en.wikipedia.org/wiki/Molecular_mass',
        'value': 28.02,
        'unit': 'g / mole',
        'min': -np.inf,
        'max': np.inf,
        'fixed': True,
    },
}
DEFAULTS.update(MATERIAL_DEFAULTS)


class MaterialDensity(Material):
    def __init__(
        self,
        chemical_structure: Union[str, None] = None,
        density: Union[Parameter, float, None] = None,
        name: str = 'EasyMaterialDensity',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        chemical_structure : Union[str, None], optional
            Chemical formula for the material. By default, None.
        density : Union[Parameter, float, None], optional
            Mass density for the material. By default, None.
        name : str, optional
            Identifier. By default, 'EasyMaterialDensity'.
        interface :
            Interface object. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if chemical_structure is None:
            chemical_structure = DEFAULTS['chemical_structure']

        density = get_as_parameter(
            name='density',
            value=density,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Density',
        )

        scattering_length = neutron_scattering_length(chemical_structure)

        mw = get_as_parameter(
            name='molecular_weight',
            value=compute_molecular_weight(chemical_structure),
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Mw',
        )
        scattering_length_real = get_as_parameter(
            name='scattering_length_real',
            value=scattering_length.real,
            default_dict=DEFAULTS['sld'],
            unique_name_prefix=f'{unique_name}_ScatteringLengthReal',
        )
        scattering_length_imag = get_as_parameter(
            name='scattering_length_imag',
            value=scattering_length.imag,
            default_dict=DEFAULTS['isld'],
            unique_name_prefix=f'{unique_name}_ScatteringLengthImag',
        )
        sld = get_as_parameter(
            name='sld',
            value=density_to_sld(scattering_length_real.value, mw.value, density.value),
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Sld',
        )
        isld = get_as_parameter(
            name='isld',
            value=density_to_sld(scattering_length_imag.value, mw.value, density.value),
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Isld',
        )

        dependency_expression = '1e-23*(0.602214076e6 * d * sl) / mw'
        dependency_map = {'d': density, 'sl': scattering_length_real, 'mw': mw}
        sld.make_dependent_on(dependency_expression=dependency_expression, dependency_map=dependency_map)

        dependency_map = {'d': density, 'sl': scattering_length_imag, 'mw': mw}
        isld.make_dependent_on(dependency_expression=dependency_expression, dependency_map=dependency_map)

        super().__init__(sld=sld, isld=isld, name=name, unique_name=unique_name, interface=None)

        self._scattering_length_real = scattering_length_real
        self._scattering_length_imag = scattering_length_imag
        self._molecular_weight = mw
        self._density = density
        self._chemical_structure = chemical_structure

        if interface is not None:
            self.interface = interface

    def _setup_sld_constraints(self) -> None:
        """Wire the derived `sld` / `isld` to depend on the current density and
        scattering-length Parameters.

        Idempotent — invoked once from `__init__` and again from `from_dict`
        after :class:`ModelBase` has swapped in the saved Parameter objects.
        """
        for derived in (self._sld, self._isld):
            if not derived.independent:
                derived.make_independent()

        dependency_expression = '1e-23*(0.602214076e6 * d * sl) / mw'
        self._sld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={
                'd': self._density,
                'sl': self._scattering_length_real,
                'mw': self._molecular_weight,
            },
        )
        self._isld.make_dependent_on(
            dependency_expression=dependency_expression,
            dependency_map={
                'd': self._density,
                'sl': self._scattering_length_imag,
                'mw': self._molecular_weight,
            },
        )

    @classmethod
    def from_dict(cls, obj_dict: dict) -> 'MaterialDensity':
        """Re-attach sld/isld dependencies after deserialization.

        :class:`ModelBase.from_dict` re-points `self._density` at the
        deserialized Parameter (because `density` is a constructor argument);
        the constraint graph built in `__init__` still references the
        temporary Parameter created from the float kwarg. Rebuild here so
        `q.density = X` propagates to the derived SLDs.
        """
        instance = super().from_dict(obj_dict)
        instance._setup_sld_constraints()
        return instance

    @property
    def chemical_structure(self) -> str:
        """Get the chemical structure string."""
        return self._chemical_structure

    @chemical_structure.setter
    def chemical_structure(self, structure_string: str) -> None:
        """Set the chemical structure string.

        Parameters
        ----------
        structure_string : str
            String that defines the chemical structure.
        """
        self._chemical_structure = structure_string
        scattering_length = neutron_scattering_length(structure_string)
        # Update the molar mass alongside the scattering length, otherwise the
        # derived SLD (d * b * N_A / M) mixes the new element's scattering length
        # with the previous element's molar mass (see issue #369).
        self._molecular_weight.value = compute_molecular_weight(structure_string)
        self._scattering_length_real.value = scattering_length.real
        self._scattering_length_imag.value = scattering_length.imag

    @property
    def density(self) -> Parameter:
        return self._density

    @density.setter
    def density(self, value: float) -> None:
        self._density.value = value

    @property
    def molecular_weight(self) -> Parameter:
        return self._molecular_weight

    @property
    def scattering_length_real(self) -> Parameter:
        return self._scattering_length_real

    @property
    def scattering_length_imag(self) -> Parameter:
        return self._scattering_length_imag

    @property
    def _dict_repr(self) -> dict[str, str]:
        """Dictionary representation of the instance."""
        mat_dict = super()._dict_repr
        mat_dict['chemical_structure'] = self._chemical_structure
        mat_dict['density'] = f'{self.density.value:.2e} {self.density.unit}'
        return mat_dict
