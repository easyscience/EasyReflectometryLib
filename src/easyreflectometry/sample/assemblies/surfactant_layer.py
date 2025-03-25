from __future__ import annotations

from typing import Optional

from easyscience import global_object
from easyscience.Constraints import ObjConstraint
from easyscience.Objects.new_variable import Parameter

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer_area_per_molecule import LayerAreaPerMolecule
from ..elements.materials.material import Material
from .base_assembly import BaseAssembly


class SurfactantLayer(BaseAssembly):
    """A surfactant layer constructs a series of layers representing the
    head and tail groups of a surfactant.
    This assembly allows the definition of a surfactant or lipid using the chemistry
    of the head (head_layer) and tail (tail_layer) regions, additionally
    this approach will make the application of constraints such as conformal roughness
    or area per molecule more straight forward.

    More information about the usage of this assembly is available in the
    `surfactant documentation`_

    .. _`surfactant documentation`: ../sample/assemblies_library.html#surfactantlayer
    """

    def __init__(
        self,
        tail_layer: Optional[LayerAreaPerMolecule] = None,
        head_layer: Optional[LayerAreaPerMolecule] = None,
        name: str = 'EasySurfactantLayer',
        unique_name: Optional[str] = None,
        constrain_area_per_molecule: bool = False,
        conformal_roughness: bool = False,
        interface=None,
    ):
        """Constructor.

        :param tail_layer: Layer representing the tail part of the surfactant layer.
        :param head_layer: Layer representing the head part of the surfactant layer.
        :param name: Name for surfactant layer, defaults to 'EasySurfactantLayer'.
        :param constrain_area_per_molecule: Constrain the area per molecule, defaults to `False`.
        :param conformal_roughness: Constrain the roughness to be the same for both layers, defaults to `False`.
        :param interface: Calculator interface, defaults to `None`.
        """
        # We need to generate a unique name to create the nested objects
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        if tail_layer is None:
            air = Material(
                sld=0,
                isld=0,
                name='Air',
                unique_name=unique_name + '_MaterialTail',
                interface=interface,
            )
            tail_layer = LayerAreaPerMolecule(
                molecular_formula='C32D64',
                thickness=16,
                solvent=air,
                solvent_fraction=0.0,
                area_per_molecule=48.2,
                roughness=3,
                name='DPPC Tail',
                unique_name=unique_name + '_LayerAreaPerMoleculeTail',
                interface=interface,
            )
        if head_layer is None:
            d2o = Material(
                sld=6.36,
                isld=0,
                name='D2O',
                unique_name=unique_name + '_MaterialHead',
                interface=interface,
            )
            head_layer = LayerAreaPerMolecule(
                molecular_formula='C10H18NO8P',
                thickness=10.0,
                solvent=d2o,
                solvent_fraction=0.2,
                area_per_molecule=48.2,
                roughness=3.0,
                name='DPPC Head',
                unique_name=unique_name + '_LayerAreaPerMoleculeHead',
                interface=interface,
            )
        surfactant = LayerCollection(
            tail_layer,
            head_layer,
            name='Layers',
            unique_name=unique_name + '_LayerCollection',
            interface=interface,
        )
        super().__init__(
            name=name,
            unique_name=unique_name,
            type='Surfactant Layer',
            layers=surfactant,
            interface=interface,
        )

        self.interface = interface
        self.head_layer._area_per_molecule.enabled = True
        area_per_molecule = ObjConstraint(
            dependent_obj=self.head_layer._area_per_molecule,
            operator='',
            independent_obj=self.tail_layer._area_per_molecule,
        )
        self.tail_layer._area_per_molecule.user_constraints['area_per_molecule'] = area_per_molecule
        self.tail_layer._area_per_molecule.user_constraints['area_per_molecule'].enabled = constrain_area_per_molecule

        self._setup_roughness_constraints()
        if conformal_roughness:
            self._enable_roughness_constraints()

    @property
    def tail_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the tail layer of the surfactant surface."""
        return self.front_layer

    @tail_layer.setter
    def tail_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the tail layer of the surfactant surface."""
        self.front_layer = layer

    @property
    def head_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the head layer of the surfactant surface."""
        return self.back_layer

    @head_layer.setter
    def head_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the head layer of the surfactant surface."""
        self.back_layer = layer

    @property
    def constrain_area_per_molecule(self) -> bool:
        """Get the area per molecule constraint status."""
        return self.tail_layer._area_per_molecule.user_constraints['area_per_molecule'].enabled

    @constrain_area_per_molecule.setter
    def constrain_area_per_molecule(self, status: bool):
        """Set the status for the area per molecule constraint such that the head and tail layers have the
        same area per molecule.

        :param status: Boolean description the wanted of the constraint.
        """
        self.tail_layer._area_per_molecule.user_constraints['area_per_molecule'].enabled = status
        if status:
            # Apply the constraint by running it
            self.tail_layer._area_per_molecule.user_constraints['area_per_molecule']()

    @property
    def conformal_roughness(self) -> bool:
        """Get the roughness constraint status."""
        return self.tail_layer.roughness.user_constraints['roughness_1'].enabled

    @conformal_roughness.setter
    def conformal_roughness(self, status: bool):
        """Set the status for the roughness to be the same for both layers.

        :param status: Boolean description the wanted of the constraint.
        """
        if status:
            self._enable_roughness_constraints()
        else:
            self._disable_roughness_constraints()

    def constrain_solvent_roughness(self, solvent_roughness: Parameter):
        """Add the constraint to the solvent roughness.

        :param solvent_roughness: The solvent roughness parameter.
        """
        if not self.conformal_roughness:
            raise ValueError('Roughness must be conformal to use this function.')
        solvent_roughness.value = self.tail_layer.roughness.value
        rough = ObjConstraint(solvent_roughness, '', self.tail_layer.roughness)
        self.tail_layer.roughness.user_constraints['solvent_roughness'] = rough

    def constain_multiple_contrast(
        self,
        another_contrast: SurfactantLayer,
        head_layer_thickness: bool = True,
        tail_layer_thickness: bool = True,
        head_layer_area_per_molecule: bool = True,
        tail_layer_area_per_molecule: bool = True,
        head_layer_fraction: bool = True,
        tail_layer_fraction: bool = True,
    ):
        """Constrain structural parameters between surfactant layer objects.

        :param another_contrast: The surfactant layer to constrain
        """
        if head_layer_thickness:
            head_layer_thickness_constraint = ObjConstraint(
                dependent_obj=self.head_layer.thickness,
                operator='',
                independent_obj=another_contrast.head_layer.thickness,
            )
            another_contrast.head_layer.thickness.user_constraints[f'{another_contrast.name}'] = (
                head_layer_thickness_constraint
            )
        if tail_layer_thickness:
            tail_layer_thickness_constraint = ObjConstraint(
                dependent_obj=self.tail_layer.thickness, operator='', independent_obj=another_contrast.tail_layer.thickness
            )
            another_contrast.tail_layer.thickness.user_constraints[f'{another_contrast.name}'] = (
                tail_layer_thickness_constraint
            )
        if head_layer_area_per_molecule:
            head_layer_area_per_molecule_constraint = ObjConstraint(
                dependent_obj=self.head_layer._area_per_molecule,
                operator='',
                independent_obj=another_contrast.head_layer._area_per_molecule,
            )
            another_contrast.head_layer._area_per_molecule.user_constraints[f'{another_contrast.name}'] = (
                head_layer_area_per_molecule_constraint
            )
        if tail_layer_area_per_molecule:
            tail_layer_area_per_molecule_constraint = ObjConstraint(
                dependent_obj=self.tail_layer._area_per_molecule,
                operator='',
                independent_obj=another_contrast.tail_layer._area_per_molecule,
            )
            another_contrast.tail_layer._area_per_molecule.user_constraints[f'{another_contrast.name}'] = (
                tail_layer_area_per_molecule_constraint
            )
        if head_layer_fraction:
            head_layer_fraction_constraint = ObjConstraint(
                dependent_obj=self.head_layer.material._fraction,
                operator='',
                independent_obj=another_contrast.head_layer.material._fraction,
            )
            another_contrast.head_layer.material._fraction.user_constraints[f'{another_contrast.name}'] = (
                head_layer_fraction_constraint
            )
        if tail_layer_fraction:
            tail_layer_fraction_constraint = ObjConstraint(
                dependent_obj=self.tail_layer.material._fraction,
                operator='',
                independent_obj=another_contrast.tail_layer.material._fraction,
            )
            another_contrast.tail_layer.material._fraction.user_constraints[f'{another_contrast.name}'] = (
                tail_layer_fraction_constraint
            )

    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        return {
            self.name: {
                'head_layer': self.head_layer._dict_repr,
                'tail_layer': self.tail_layer._dict_repr,
                'area per molecule constrained': self.constrain_area_per_molecule,
                'conformal roughness': self.conformal_roughness,
            }
        }

    def as_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Produces a cleaned dict using a custom as_dict method to skip necessary things.
        The resulting dict matches the parameters in __init__

        :param skip: List of keys to skip, defaults to `None`.
        """
        this_dict = super().as_dict(skip=skip)
        this_dict['tail_layer'] = self.tail_layer.as_dict(skip=skip)
        this_dict['head_layer'] = self.head_layer.as_dict(skip=skip)
        this_dict['constrain_area_per_molecule'] = self.constrain_area_per_molecule
        this_dict['conformal_roughness'] = self.conformal_roughness
        del this_dict['layers']
        return this_dict
