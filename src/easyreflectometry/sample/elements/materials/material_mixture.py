from typing import Optional
from typing import Union

from easyscience import global_object
from easyscience.Constraints import FunctionalConstraint
from easyscience.Objects.new_variable import Parameter

from easyreflectometry.special.calculations import weighted_average
from easyreflectometry.utils import get_as_parameter

from ...base_core import BaseCore
from .material import DEFAULTS as MATERIAL_DEFAULTS
from .material import Material

DEFAULTS = {
    'fraction': {
        'description': 'The fraction of material b in material a',
        'value': 0.5,
        'unit': 'dimensionless',
        'min': 0,
        'max': 1,
        'fixed': True,
    }
}
DEFAULTS.update(MATERIAL_DEFAULTS)


class MaterialMixture(BaseCore):
    # Added in super().__init__
    _material_a: Material
    _material_b: Material
    _fraction: Parameter

    def __init__(
        self,
        material_a: Union[Material, None] = None,
        material_b: Union[Material, None] = None,
        fraction: Union[Parameter, float, None] = None,
        name: Union[str, None] = None,
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        :param material_a: The first material.
        :param material_b: The second material.
        :param fraction: The fraction of material_b in material_a.
        :param name: Name of the material, defaults to None that causes the name to be constructed.
        :param interface: Calculator interface, defaults to `None`.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if material_a is None:
            material_a = Material(interface=interface)
        if material_b is None:
            material_b = Material(interface=interface)

        fraction = get_as_parameter(
            name='fraction',
            value=fraction,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Fraction',
        )

        sld = weighted_average(
            a=material_a.sld.value,
            b=material_b.sld.value,
            p=fraction.value,
        )
        isld = weighted_average(
            a=material_a.isld.value,
            b=material_b.isld.value,
            p=fraction.value,
        )

        self._sld = get_as_parameter(
            name='sld',
            value=sld,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Sld',
        )
        self._isld = get_as_parameter(
            name='isld',
            value=isld,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Isld',
        )

        # To avoid problems when setting the interface
        # self._sld and self._isld need to be declared before calling the super constructor
        super().__init__(
            name,
            _material_a=material_a,
            _material_b=material_b,
            _fraction=fraction,
            interface=interface,
        )
        if name is None:
            self._update_name()

        self._materials_constraints()
        self.interface = interface

    def _get_linkable_attributes(self):
        return [self._sld, self._isld]

    @property
    def sld(self) -> float:
        return self._sld.value

    @property
    def isld(self) -> float:
        return self._isld.value

    def _materials_constraints(self):
        self._sld.enabled = True
        self._isld.enabled = True
        constraint = FunctionalConstraint(
            dependent_obj=self._sld,
            func=weighted_average,
            independent_objs=[self._material_a.sld, self._material_b.sld, self._fraction],
        )
        self._material_a.sld.user_constraints['sld'] = constraint
        self._material_b.sld.user_constraints['sld'] = constraint
        self._fraction.user_constraints['sld'] = constraint
        constraint()
        iconstraint = FunctionalConstraint(
            dependent_obj=self._isld,
            func=weighted_average,
            independent_objs=[self._material_a.isld, self._material_b.isld, self._fraction],
        )
        self._material_a.isld.user_constraints['isld'] = iconstraint
        self._material_b.isld.user_constraints['isld'] = iconstraint
        self._fraction.user_constraints['isld'] = iconstraint
        iconstraint()

    @property
    def fraction(self) -> float:
        """Get the fraction of material_b."""
        return self._fraction.value

    @fraction.setter
    def fraction(self, fraction: float) -> None:
        """Setter for fraction of material_b.

        :param fraction: The fraction of material_b in material_a.
        """
        if not isinstance(fraction, float):
            raise ValueError('fraction must be a float')
        self._fraction.value = fraction

    @property
    def material_a(self) -> Material:
        """Getter for material_a."""
        return self._material_a

    @material_a.setter
    def material_a(self, new_material_a: Material) -> None:
        """Setter for material_a

        :param new_material_a: New Material for material_a
        """
        self._material_a = new_material_a
        self._materials_constraints()
        if self.interface is not None:
            self.interface.generate_bindings(self)
        self._update_name()

    @property
    def material_b(self) -> Material:
        """Getter for material_b."""
        return self._material_b

    @material_b.setter
    def material_b(self, new_material_b: Material) -> None:
        """Setter for material_b

        :param new_material_b: New Materialfor material_b
        """
        self._material_b = new_material_b
        self._materials_constraints()
        if self.interface is not None:
            self.interface.generate_bindings(self)
        self._update_name()

    def _update_name(self) -> None:
        self.name = self._material_a.name + '/' + self._material_b.name

    # Representation
    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            self.name: {
                'fraction': f'{self._fraction.value:.3f} {self._fraction.unit}',
                'sld': f'{self._sld.value:.3f}e-6 {self._sld.unit}',
                'isld': f'{self._isld.value:.3f}e-6 {self._isld.unit}',
                'material_a': self._material_a._dict_repr,
                'material_b': self._material_b._dict_repr,
            }
        }

    def as_dict(self, skip: Optional[list[str]] = None) -> dict[str, str]:
        """Produces a cleaned dict using a custom as_dict method to skip necessary things.
        The resulting dict matches the parameters in __init__

        :param skip: List of keys to skip, defaults to `None`.
        """
        this_dict = super().as_dict(skip=skip)
        this_dict['material_a'] = self._material_a.as_dict(skip=skip)
        this_dict['material_b'] = self._material_b.as_dict(skip=skip)
        this_dict['fraction'] = self._fraction.as_dict(skip=skip)
        return this_dict
