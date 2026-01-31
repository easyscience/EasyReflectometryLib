from __future__ import annotations

from typing import Optional

from easyscience import global_object
from easyscience.variable import Parameter

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer_area_per_molecule import LayerAreaPerMolecule
from ..elements.materials.material import Material
from .base_assembly import BaseAssembly


class Bilayer(BaseAssembly):
    """A lipid bilayer consisting of two surfactant layers where one is inverted.

    The bilayer structure is: Front Head - Front Tail - Back Tail - Back Head

    This assembly comes pre-populated with physically meaningful constraints:
    - Both tail layers share the same structural parameters (thickness, area per molecule)
    - Head layers share thickness and area per molecule (different hydration/solvent_fraction allowed)
    - A single roughness parameter applies to all interfaces (conformal roughness)

    More information about the usage of this assembly is available in the
    `bilayer documentation`_

    .. _`bilayer documentation`: ../sample/assemblies_library.html#bilayer
    """

    def __init__(
        self,
        front_head_layer: Optional[LayerAreaPerMolecule] = None,
        tail_layer: Optional[LayerAreaPerMolecule] = None,
        back_head_layer: Optional[LayerAreaPerMolecule] = None,
        name: str = 'EasyBilayer',
        unique_name: Optional[str] = None,
        constrain_heads: bool = True,
        conformal_roughness: bool = True,
        interface=None,
    ):
        """Constructor.

        :param front_head_layer: Layer representing the front head part of the bilayer.
        :param tail_layer: Layer representing the tail part of the bilayer. A second tail
            layer is created internally with parameters constrained to this one.
        :param back_head_layer: Layer representing the back head part of the bilayer.
        :param name: Name for bilayer, defaults to 'EasyBilayer'.
        :param constrain_heads: Constrain head layer thickness and area per molecule, defaults to `True`.
        :param conformal_roughness: Constrain roughness to be the same for all layers, defaults to `True`.
        :param interface: Calculator interface, defaults to `None`.
        """
        # Generate unique name for nested objects
        if unique_name is None:
            unique_name = global_object.generate_unique_name(self.__class__.__name__)

        # Create default front head layer if not provided
        if front_head_layer is None:
            d2o_front = Material(
                sld=6.36,
                isld=0,
                name='D2O',
                unique_name=unique_name + '_MaterialFrontHead',
                interface=interface,
            )
            front_head_layer = LayerAreaPerMolecule(
                molecular_formula='C10H18NO8P',
                thickness=10.0,
                solvent=d2o_front,
                solvent_fraction=0.2,
                area_per_molecule=48.2,
                roughness=3.0,
                name='DPPC Head Front',
                unique_name=unique_name + '_LayerAreaPerMoleculeFrontHead',
                interface=interface,
            )

        # Create default tail layer if not provided
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
                thickness=16.0,
                solvent=air,
                solvent_fraction=0.0,
                area_per_molecule=48.2,
                roughness=3.0,
                name='DPPC Tail',
                unique_name=unique_name + '_LayerAreaPerMoleculeTail',
                interface=interface,
            )

        # Create second tail layer with same parameters as the first
        # This will be constrained to the first tail layer
        air_back = Material(
            sld=0,
            isld=0,
            name='Air',
            unique_name=unique_name + '_MaterialBackTail',
            interface=interface,
        )
        back_tail_layer = LayerAreaPerMolecule(
            molecular_formula=tail_layer.molecular_formula,
            thickness=tail_layer.thickness.value,
            solvent=air_back,
            solvent_fraction=tail_layer.solvent_fraction,
            area_per_molecule=tail_layer.area_per_molecule,
            roughness=tail_layer.roughness.value,
            name=tail_layer.name + ' Back',
            unique_name=unique_name + '_LayerAreaPerMoleculeBackTail',
            interface=interface,
        )

        # Create default back head layer if not provided
        if back_head_layer is None:
            d2o_back = Material(
                sld=6.36,
                isld=0,
                name='D2O',
                unique_name=unique_name + '_MaterialBackHead',
                interface=interface,
            )
            back_head_layer = LayerAreaPerMolecule(
                molecular_formula='C10H18NO8P',
                thickness=10.0,
                solvent=d2o_back,
                solvent_fraction=0.2,
                area_per_molecule=48.2,
                roughness=3.0,
                name='DPPC Head Back',
                unique_name=unique_name + '_LayerAreaPerMoleculeBackHead',
                interface=interface,
            )

        # Store the front tail layer reference for constraint setup
        self._front_tail_layer = tail_layer
        self._back_tail_layer = back_tail_layer

        # Create layer collection: front_head, front_tail, back_tail, back_head
        bilayer_layers = LayerCollection(
            front_head_layer,
            tail_layer,
            back_tail_layer,
            back_head_layer,
            name='Layers',
            unique_name=unique_name + '_LayerCollection',
            interface=interface,
        )

        super().__init__(
            name=name,
            unique_name=unique_name,
            type='Bilayer',
            layers=bilayer_layers,
            interface=interface,
        )

        self.interface = interface
        self._conformal_roughness = False
        self._constrain_heads = False
        self._tail_constraints_setup = False

        # Setup tail layer constraints (back tail depends on front tail)
        self._setup_tail_constraints()

        # Apply head constraints if requested
        if constrain_heads:
            self.constrain_heads = True

        # Apply conformal roughness if requested
        if conformal_roughness:
            self.conformal_roughness = True

    def _setup_tail_constraints(self) -> None:
        """Setup constraints so back tail layer parameters depend on front tail layer."""
        front_tail = self._front_tail_layer
        back_tail = self._back_tail_layer

        # Constrain thickness
        back_tail.thickness.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_tail.thickness},
        )

        # Constrain area per molecule
        back_tail._area_per_molecule.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_tail._area_per_molecule},
        )

        # Constrain solvent fraction
        back_tail.material._fraction.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_tail.material._fraction},
        )

        self._tail_constraints_setup = True

    @property
    def front_head_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the front head layer of the bilayer."""
        return self.layers[0]

    @front_head_layer.setter
    def front_head_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the front head layer of the bilayer."""
        self.layers[0] = layer

    @property
    def front_tail_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the front tail layer of the bilayer."""
        return self.layers[1]

    @property
    def tail_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the tail layer (alias for front_tail_layer for serialization compatibility)."""
        return self.front_tail_layer

    @property
    def back_tail_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the back tail layer of the bilayer."""
        return self.layers[2]

    @property
    def back_head_layer(self) -> Optional[LayerAreaPerMolecule]:
        """Get the back head layer of the bilayer."""
        return self.layers[3]

    @back_head_layer.setter
    def back_head_layer(self, layer: LayerAreaPerMolecule) -> None:
        """Set the back head layer of the bilayer."""
        self.layers[3] = layer

    @property
    def constrain_heads(self) -> bool:
        """Get the head layer constraint status."""
        return self._constrain_heads

    @constrain_heads.setter
    def constrain_heads(self, status: bool) -> None:
        """Set the status for head layer constraints.

        When enabled, the back head layer thickness and area per molecule
        are constrained to match the front head layer. Solvent fraction
        (hydration) remains independent.

        :param status: Boolean for the constraint status.
        """
        if status:
            self._enable_head_constraints()
        else:
            self._disable_head_constraints()
        self._constrain_heads = status

    def _enable_head_constraints(self) -> None:
        """Enable head layer constraints (thickness and area per molecule)."""
        front_head = self.front_head_layer
        back_head = self.back_head_layer

        # Constrain thickness
        back_head.thickness.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_head.thickness},
        )

        # Constrain area per molecule
        back_head._area_per_molecule.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': front_head._area_per_molecule},
        )

    def _disable_head_constraints(self) -> None:
        """Disable head layer constraints."""
        self.back_head_layer.thickness.make_independent()
        self.back_head_layer._area_per_molecule.make_independent()

    @property
    def conformal_roughness(self) -> bool:
        """Get the roughness constraint status."""
        return self._conformal_roughness

    @conformal_roughness.setter
    def conformal_roughness(self, status: bool) -> None:
        """Set the status for conformal roughness.

        When enabled, all layers share the same roughness parameter
        (controlled by the front head layer).

        :param status: Boolean for the constraint status.
        """
        if status:
            self._setup_roughness_constraints()
            self._enable_roughness_constraints()
        else:
            if self._roughness_constraints_setup:
                self._disable_roughness_constraints()
        self._conformal_roughness = status

    def constrain_solvent_roughness(self, solvent_roughness: Parameter) -> None:
        """Add the constraint to the solvent roughness.

        :param solvent_roughness: The solvent roughness parameter.
        """
        if not self.conformal_roughness:
            raise ValueError('Roughness must be conformal to use this function.')
        solvent_roughness.value = self.front_head_layer.roughness.value
        solvent_roughness.make_dependent_on(
            dependency_expression='a',
            dependency_map={'a': self.front_head_layer.roughness},
        )

    def constrain_multiple_contrast(
        self,
        another_contrast: Bilayer,
        front_head_thickness: bool = True,
        back_head_thickness: bool = True,
        tail_thickness: bool = True,
        front_head_area_per_molecule: bool = True,
        back_head_area_per_molecule: bool = True,
        tail_area_per_molecule: bool = True,
        front_head_fraction: bool = True,
        back_head_fraction: bool = True,
        tail_fraction: bool = True,
    ) -> None:
        """Constrain structural parameters between bilayer objects.

        :param another_contrast: The bilayer to constrain to.
        :param front_head_thickness: Constrain front head thickness.
        :param back_head_thickness: Constrain back head thickness.
        :param tail_thickness: Constrain tail thickness.
        :param front_head_area_per_molecule: Constrain front head area per molecule.
        :param back_head_area_per_molecule: Constrain back head area per molecule.
        :param tail_area_per_molecule: Constrain tail area per molecule.
        :param front_head_fraction: Constrain front head solvent fraction.
        :param back_head_fraction: Constrain back head solvent fraction.
        :param tail_fraction: Constrain tail solvent fraction.
        """
        if front_head_thickness:
            self.front_head_layer.thickness.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.front_head_layer.thickness},
            )

        if back_head_thickness:
            self.back_head_layer.thickness.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.back_head_layer.thickness},
            )

        if tail_thickness:
            self.front_tail_layer.thickness.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.front_tail_layer.thickness},
            )

        if front_head_area_per_molecule:
            self.front_head_layer._area_per_molecule.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.front_head_layer._area_per_molecule},
            )

        if back_head_area_per_molecule:
            self.back_head_layer._area_per_molecule.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.back_head_layer._area_per_molecule},
            )

        if tail_area_per_molecule:
            self.front_tail_layer._area_per_molecule.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.front_tail_layer._area_per_molecule},
            )

        if front_head_fraction:
            self.front_head_layer.material._fraction.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.front_head_layer.material._fraction},
            )

        if back_head_fraction:
            self.back_head_layer.material._fraction.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.back_head_layer.material._fraction},
            )

        if tail_fraction:
            self.front_tail_layer.material._fraction.make_dependent_on(
                dependency_expression='a',
                dependency_map={'a': another_contrast.front_tail_layer.material._fraction},
            )

    @property
    def _dict_repr(self) -> dict:
        """A simplified dict representation."""
        return {
            self.name: {
                'front_head_layer': self.front_head_layer._dict_repr,
                'front_tail_layer': self.front_tail_layer._dict_repr,
                'back_tail_layer': self.back_tail_layer._dict_repr,
                'back_head_layer': self.back_head_layer._dict_repr,
                'constrain_heads': self.constrain_heads,
                'conformal_roughness': self.conformal_roughness,
            }
        }

    def as_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Produce a cleaned dict using a custom as_dict method.

        The resulting dict matches the parameters in __init__

        :param skip: List of keys to skip, defaults to `None`.
        """
        this_dict = super().as_dict(skip=skip)
        this_dict['front_head_layer'] = self.front_head_layer.as_dict(skip=skip)
        this_dict['tail_layer'] = self.front_tail_layer.as_dict(skip=skip)
        this_dict['back_head_layer'] = self.back_head_layer.as_dict(skip=skip)
        this_dict['constrain_heads'] = self.constrain_heads
        this_dict['conformal_roughness'] = self.conformal_roughness
        del this_dict['layers']
        return this_dict
