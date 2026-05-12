# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

from easyscience import global_object
from easyscience.variable import Parameter

from easyreflectometry.utils import get_as_parameter

from .material import Material
from .material_mixture import MaterialMixture

DEFAULTS = {
    'solvent_fraction': {
        'description': 'Fraction of solvent in layer.',
        'value': 0.2,
        'unit': 'dimensionless',
        'min': 0,
        'max': 1,
        'fixed': True,
    },
}


class MaterialSolvated(MaterialMixture):
    def __init__(
        self,
        material: Union[Material, None] = None,
        solvent: Union[Material, None] = None,
        solvent_fraction: Union[Parameter, float, None] = None,
        name=None,
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        material : Union[Material, None], optional
            The material being solvated. By default, None.
        solvent : Union[Material, None], optional
            The solvent material. By default, None.
        solvent_fraction : Union[Parameter, float, None], optional
            Fraction of solvent in layer. E.g. solvation or surface coverage. By default, None.
        name :
            Name of the material. By default, None.
        interface :
            Calculator interface. By default, None.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if material is None:
            material = Material(sld=6.36, isld=0, name='D2O', interface=interface)
        if solvent is None:
            solvent = Material(sld=-0.561, isld=0, name='H2O', interface=interface)

        solvent_fraction = get_as_parameter(
            name='solvent_fraction',
            value=solvent_fraction,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Fraction',
        )

        # In super class, the fraction is the fraction of material b in material a
        super().__init__(
            material_a=material,
            material_b=solvent,
            fraction=solvent_fraction,
            name=name,
            interface=interface,
        )
        if name is None:
            self._update_name()

    @property
    def material(self) -> Material:
        """Get material."""
        return self._material_a

    @material.setter
    def material(self, new_material: Material) -> None:
        """Set the material.

        Parameters
        ----------
        new_material : Material
            Matrerial to be useed.
        """
        self.material_a = new_material

    @property
    def solvent(self) -> Material:
        """Get solvent."""
        return self._material_b

    @solvent.setter
    def solvent(self, new_solvent: Material) -> None:
        """Set the solvent.

        Parameters
        ----------
        new_solvent : Material
            Solvent to be used.
        """
        self.material_b = new_solvent

    @property
    def solvent_fraction_parameter(self) -> Parameter:
        """Get the parameter for the fraction of layer described by the solvent."""
        return self._fraction

    @property
    def solvent_fraction(self) -> float:
        """Get the fraction of layer described by the solvent.

        This might be fraction of:
        Solvation where solvent is within the layer
        Patches of solvent in the layer where no material is present.
        """
        return self.fraction

    @solvent_fraction.setter
    def solvent_fraction(self, solvent_fraction: float) -> None:
        """Set the fraction of layer covered by the material.

        This might be fraction of:
        Solvation where solvent is within the layer
        Patches of solvent in the layer where no material is present.

        Parameters
        ----------
        solvent_fraction : float
            Fraction of layer described by the solvent.
        """
        try:
            self.fraction = solvent_fraction
            if solvent_fraction < 0 or solvent_fraction > 1:
                raise ValueError('solvent_fraction must be between 0 and 1')
        except ValueError:
            raise ValueError('solvent_fraction must be a float between 0 and 1')

    def _update_name(self) -> None:
        """Update name."""
        self.name = self._material_a.name + ' in ' + self._material_b.name

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            self.name: {
                'solvent_fraction': f'{self._fraction.value:.3f} {self._fraction.unit}',
                'sld': f'{self._sld.value:.3f}e-6 {self._sld.unit}',
                'isld': f'{self._isld.value:.3f}e-6 {self._isld.unit}',
                'material': self.material._dict_repr,
                'solvent': self.solvent._dict_repr,
            }
        }

    def as_dict(self, skip: Optional[list[str]] = None) -> dict[str, str]:
        """Produces a cleaned dict using a custom as_dict method to skip necessary things.

        The resulting dict matches the parameters in __init__

        Parameters
        ----------
        skip : Optional[list[str]], optional
            List of keys to skip. By default, None.
        """
        this_dict = super().as_dict(skip=skip)
        this_dict['material'] = self.material.as_dict(skip=skip)
        this_dict['solvent'] = self.solvent.as_dict(skip=skip)
        this_dict['solvent_fraction'] = self._fraction.as_dict(skip=skip)
        # Property and protected varible from material_mixture
        del this_dict['material_a']
        del this_dict['_material_a']
        # Property and protected varible from material_mixture
        del this_dict['material_b']
        del this_dict['_material_b']
        # Property and protected varible from material_mixture
        del this_dict['fraction']
        del this_dict['_fraction']
        return this_dict
