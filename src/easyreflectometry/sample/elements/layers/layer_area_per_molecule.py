from typing import Optional
from typing import Union

import numpy as np
from easyscience import global_object
from easyscience.Constraints import FunctionalConstraint
from easyscience.Objects.new_variable import Parameter

from easyreflectometry.special.calculations import area_per_molecule_to_scattering_length_density
from easyreflectometry.special.calculations import neutron_scattering_length
from easyreflectometry.utils import get_as_parameter

from ..materials.material import Material
from ..materials.material_solvated import DEFAULTS as MATERIAL_SOLVATED_DEFAULTS
from ..materials.material_solvated import MaterialSolvated
from .layer import DEFAULTS as LAYER_DEFAULTS
from .layer import Layer

DEFAULTS = {
    'molecular_formula': 'C10H18NO8P',
    'area_per_molecule': {
        'description': 'Surface coverage',
        'value': 48.2,
        'unit': 'angstrom^2',
        'min': 0,
        'max': np.inf,
        'fixed': True,
    },
    'sl': {
        'description': 'The real scattering length for a molecule formula in angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 4.186,
        'unit': 'angstrom',
        'min': -np.Inf,
        'max': np.Inf,
        'fixed': True,
    },
    'isl': {
        'description': 'The real scattering length for a molecule formula in angstrom.',
        'url': 'https://www.ncnr.nist.gov/resources/activation/',
        'value': 0.0,
        'unit': 'angstrom',
        'min': -np.Inf,
        'max': np.Inf,
        'fixed': True,
    },
}
DEFAULTS.update(MATERIAL_SOLVATED_DEFAULTS)
DEFAULTS.update(LAYER_DEFAULTS)


class LayerAreaPerMolecule(Layer):
    """The `LayerAreaPerMolecule` class allows a layer to be defined in terms of some
    molecular formula an area per molecule, and a solvent.

    """

    # Added in __init__
    #: Real part of the scattering length.
    _scattering_length_real: Parameter
    #: Imaginary part of the scattering length.
    _scattering_length_imag: Parameter
    #: Area per molecule in the layer in Anstrom^2.
    _area_per_molecule: Parameter

    # Other typer than in __init__.super()
    material: MaterialSolvated

    def __init__(
        self,
        molecular_formula: Union[str, None] = None,
        thickness: Union[Parameter, float, None] = None,
        solvent: Union[Material, None] = None,
        solvent_fraction: Union[Parameter, float, None] = None,
        area_per_molecule: Union[Parameter, float, None] = None,
        roughness: Union[Parameter, float, None] = None,
        name: str = 'EasyLayerAreaPerMolecule',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        :param molecular_formula: Formula for the molecule in the layer.
        :param thickness: Layer thickness in Angstrom.
        :param solvent: Solvent containing the molecule.
        :param solvent_fraction: Fraction of solvent in layer. Fx solvation or surface coverage.
        :param area_per_molecule: Area per molecule in the layer
        :param roughness: Upper roughness on the layer in Angstrom.
        :param name: Name of the layer, defaults to "EasyLayerAreaPerMolecule"
        :param interface: Interface object, defaults to `None`
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if solvent is None:
            solvent = Material(
                sld=6.36,
                isld=0,
                name='D2O',
                unique_name=unique_name + '_MaterialSolvent',
                interface=interface,
            )

        # Create the solvated molecule and corresponding constraints
        if molecular_formula is None:
            molecular_formula = DEFAULTS['molecular_formula']
        molecule_material = Material(
            sld=0.0,
            isld=0.0,
            name=molecular_formula,
            unique_name=unique_name + '_MaterialMolecule',
            interface=interface,
        )

        thickness = get_as_parameter(
            name='thickness',
            value=thickness,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_Thickness',
        )
        _area_per_molecule = get_as_parameter(
            name='area_per_molecule',
            value=area_per_molecule,
            default_dict=DEFAULTS,
            unique_name_prefix=f'{unique_name}_AreaPerMolecule',
        )
        _scattering_length_real = get_as_parameter(
            name='scattering_length_real',
            value=0.0,
            default_dict=DEFAULTS['sl'],
            unique_name_prefix=f'{unique_name}_Sl',
        )
        _scattering_length_imag = get_as_parameter(
            name='scattering_length_imag',
            value=0.0,
            default_dict=DEFAULTS['isl'],
            unique_name_prefix=f'{unique_name}_Isl',
        )

        # Constrain the real part of the sld value for the molecule
        constraint_sld_real = FunctionalConstraint(
            dependent_obj=molecule_material.sld,
            func=area_per_molecule_to_scattering_length_density,
            independent_objs=[_scattering_length_real, thickness, _area_per_molecule],
        )
        thickness.user_constraints['area_per_molecule'] = constraint_sld_real
        _area_per_molecule.user_constraints['area_per_molecule'] = constraint_sld_real
        _scattering_length_real.user_constraints['area_per_molecule'] = constraint_sld_real

        # Constrain the imaginary part of the sld value for the molecule
        constraint_sld_imag = FunctionalConstraint(
            dependent_obj=molecule_material.isld,
            func=area_per_molecule_to_scattering_length_density,
            independent_objs=[_scattering_length_imag, thickness, _area_per_molecule],
        )
        thickness.user_constraints['iarea_per_molecule'] = constraint_sld_imag
        _area_per_molecule.user_constraints['iarea_per_molecule'] = constraint_sld_imag
        _scattering_length_imag.user_constraints['iarea_per_molecule'] = constraint_sld_imag

        solvated_molecule_material = MaterialSolvated(
            material=molecule_material,
            solvent=solvent,
            solvent_fraction=solvent_fraction,
            unique_name=unique_name + '_MaterialSolvated',
            interface=interface,
        )
        super().__init__(
            material=solvated_molecule_material,
            thickness=thickness,
            roughness=roughness,
            name=name,
            unique_name=unique_name,
            interface=interface,
        )
        self._add_component('_scattering_length_real', _scattering_length_real)
        self._add_component('_scattering_length_imag', _scattering_length_imag)
        self._add_component('_area_per_molecule', _area_per_molecule)

        scattering_length = neutron_scattering_length(molecular_formula)
        self._scattering_length_real.value = scattering_length.real
        self._scattering_length_imag.value = scattering_length.imag
        self._molecular_formula = molecular_formula
        self.interface = interface

    @property
    def area_per_molecule_parameter(self) -> Parameter:
        """Get the parameter for area per molecule."""
        return self._area_per_molecule

    @property
    def area_per_molecule(self) -> float:
        """Get the area per molecule."""
        return self._area_per_molecule.value

    @area_per_molecule.setter
    def area_per_molecule(self, new_area_per_molecule: float) -> None:
        """Set the area per molecule.

        :param new_area_per_molecule: New area per molecule.
        """
        if new_area_per_molecule < 0:
            raise ValueError('new_area_per_molecule must be greater than 0.0.')
        self._area_per_molecule.value = new_area_per_molecule

    @property
    def molecule(self) -> Material:
        """Get the molecule material."""
        return self.material.material

    @property
    def solvent(self) -> Material:
        """Get the solvent material."""
        return self.material.solvent

    @solvent.setter
    def solvent(self, new_solvent: Material) -> None:
        """Set the solvent material.

        :param new_solvent: New solvent material.
        """
        self.material.solvent = new_solvent

    @property
    def solvent_fraction_parameter(self) -> float:
        """Get parameter for the fraction of the layer occupied by the solvent."""
        return self.material.solvent_fraction_parameter

    @property
    def solvent_fraction(self) -> float:
        """Get the fraction of the layer occupied by the solvent.
        This could be a result of either water solvating the molecule, or incomplete surface coverage of the molecules.
        """
        return self.material.solvent_fraction

    @solvent_fraction.setter
    def solvent_fraction(self, solvent_fraction: float) -> None:
        """Set the fraction of the layer occupied by the solvent.
        This could be a result of either water solvating the molecule, or incomplete surface coverage of the molecules.

        :param solvent_fraction: Fraction of layer described by the solvent.
        """
        self.material.solvent_fraction = solvent_fraction

    @property
    def molecular_formula(self) -> str:
        """Get the formula of molecule the layer."""
        return self._molecular_formula

    @molecular_formula.setter
    def molecular_formula(self, formula_string: str) -> None:
        """Set the formula of the molecule in the material.

        :param formula_string: String that defines the molecular formula.
        """
        self._molecular_formula = formula_string
        scattering_length = neutron_scattering_length(formula_string)
        # The molecule is also being updated through the constraints
        self._scattering_length_real.value = scattering_length.real
        self._scattering_length_imag.value = scattering_length.imag

        self.molecule.name = formula_string
        self.material._update_name()

    @property
    def _dict_repr(self) -> dict[str, str]:
        """Dictionary representation of the `area_per_molecule` object. Produces a simple dictionary"""
        dict_repr = super()._dict_repr
        dict_repr['molecular_formula'] = self._molecular_formula
        dict_repr['area_per_molecule'] = f'{self.area_per_molecule:.2f} ' f'{self._area_per_molecule.unit}'
        return dict_repr

    def as_dict(self, skip: Optional[list[str]] = None) -> dict[str, str]:
        """Produces a cleaned dict using a custom as_dict method to skip necessary things.
        The resulting dict matches the parameters in __init__

        :param skip: List of keys to skip, defaults to `None`.
        """
        this_dict = super().as_dict(skip=skip)
        this_dict['solvent_fraction'] = self.material._fraction.as_dict(skip=skip)
        this_dict['area_per_molecule'] = self._area_per_molecule.as_dict(skip=skip)
        this_dict['solvent'] = self.solvent.as_dict(skip=skip)
        del this_dict['material']
        del this_dict['_scattering_length_real']
        del this_dict['_scattering_length_imag']
        return this_dict
