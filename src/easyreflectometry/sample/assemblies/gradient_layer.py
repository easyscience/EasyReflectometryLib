# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from easyscience import global_object
from easyscience.variable import Parameter
from numpy import linspace

from ..collections.layer_collection import LayerCollection
from ..elements.layers.layer import Layer
from ..elements.materials.material import Material
from .base_assembly import BaseAssembly


class GradientLayer(BaseAssembly):
    """A set of discrete gradient layers changing from the front to the back material.

    The front layer is where the neutron beam starts in, it has an index of 0.
    """

    def __init__(
        self,
        front_material: Optional[Material] = None,
        back_material: Optional[Material] = None,
        thickness: Optional[float] = 2.0,
        roughness: Optional[float] = 0.2,
        discretisation_elements: int = 10,
        name: str = 'EasyGradienLayer',
        unique_name: Optional[str] = None,
        interface=None,
    ):
        """Constructor.

        Parameters
        ----------
        unique_name : Optional[str], optional
            By default, None.
        front_material : Optional[Material], optional
            Material of front of the layer. By default, None.
        back_material : Optional[Material], optional
            Material of back of the layer. By default, None.
        thickness : Optional[float], optional
            Thicknkess of the layer. By default, 2.0.
        roughness : Optional[float], optional
            Roughness of the layer. By default, 0.2.
        discretisation_elements : int, optional
            Number of discrete layers. By default, 10.
        name : str, optional
            Name for gradient layer. By default, 'EasyGradienLayer'.
        interface :
            Calculator interface. By default, None.
        """

        if front_material is None:
            front_material = Material(0.0, 0.0, 'Air')

        if back_material is None:
            back_material = Material(6.36, 0.0, 'D2O')

        if discretisation_elements < 2:
            raise ValueError('Discretisation elements must be greater than 2.')

        gradient_layers = _prepare_gradient_layers(
            front_material=front_material,
            back_material=back_material,
            discretisation_elements=discretisation_elements,
            interface=interface,
        )

        super().__init__(
            layers=gradient_layers,
            name=name,
            unique_name=unique_name,
            interface=None,
            type='Gradient-layer',
        )
        self._front_material = front_material
        self._back_material = back_material
        self._discretisation_elements = discretisation_elements

        self._setup_thickness_constraints()
        self._enable_thickness_constraints()
        self._setup_roughness_constraints()
        self._enable_roughness_constraints()

        # Set the thickness and roughness properties
        self.thickness = thickness
        self.roughness = roughness

        if interface is not None:
            self.interface = interface

    @property
    def front_material(self) -> Material:
        return self._front_material

    @property
    def back_material(self) -> Material:
        return self._back_material

    @property
    def discretisation_elements(self) -> int:
        return self._discretisation_elements

    @property
    def thickness(self) -> float:
        """Get the thickness of the gradient layer in Angstrom."""
        return self.front_layer.thickness.value * self._discretisation_elements

    @thickness.setter
    def thickness(self, thickness: float) -> None:
        """Set the thickness of the gradient layer.

        Parameters
        ----------
        thickness : float
            Thickness of the gradient layer in Angstroms.
        """
        self.front_layer.thickness.value = thickness / self._discretisation_elements

    @property
    def roughness(self) -> float:
        """Get the Roughness of the gradient layer in Angstrom."""
        return self.front_layer.roughness.value

    @roughness.setter
    def roughness(self, roughness: float) -> None:
        """Set the roughness of the gradient layer.

        Parameters
        ----------
        roughness : float
            Roughness of the gradient layer in Angstroms.
        """
        self.front_layer.roughness.value = roughness

    @property
    def _dict_repr(self) -> dict[str, str]:
        """A simplified dict representation."""
        return {
            'thickness': float(self.thickness),  # Conversion to float is necessary to prevent property reference in dict
            'discretisation_elements': int(self._discretisation_elements),  # Same as above
            'back_layer': self.back_layer._dict_repr,
            'front_layer': self.front_layer._dict_repr,
        }

    def to_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Produces a cleaned dict using a custom to_dict method to skip necessary things.

        The resulting dict matches the parameters in __init__: layers are derived
        in ``__init__`` from ``front_material``/``back_material``/``discretisation_elements``
        so they are excluded from the serialized representation.

        Parameters
        ----------
        skip : Optional[list[str]], optional
            List of keys to skip. By default, None.
        """
        this_dict = super().to_dict(skip=skip)
        # Determined in __init__
        this_dict.pop('layers', None)
        # `thickness` / `roughness` are read-only float views; the serialized
        # constructor args are the floats themselves.
        this_dict['thickness'] = float(self.thickness)
        this_dict['roughness'] = float(self.roughness)
        return this_dict

    def as_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Compatibility alias for :meth:`to_dict`."""
        return self.to_dict(skip=skip)


def _linear_gradient(
    front_value: float,
    back_value: float,
    discretisation_elements: int,
) -> list[float]:
    """Linear gradient of ``discretisation_elements`` values.

    Both the front and back values are included as the first and last
    elements respectively (see issue #373 — the previous implementation left
    the last sublayer one step short of the back value).
    """
    return list(linspace(front_value, back_value, discretisation_elements))


def _gradient_fraction(index: int, discretisation_elements: int) -> float:
    """Interpolation fraction (0 at the front, 1 at the back) for sublayer ``index``."""
    return index / (discretisation_elements - 1)


def _constrain_to_endpoints(
    derived: 'Parameter',
    front: 'Parameter',
    back: 'Parameter',
    fraction: float,
) -> None:
    """Make ``derived`` a linear interpolation between the ``front`` and ``back`` parameters.

    Wiring the sublayer SLDs to the end-material parameters (rather than
    snapshotting their values once) means mutating or *fitting* the front/back
    material SLDs propagates to the computed reflectivity (see issue #373).
    """
    if not derived.independent:
        derived.make_independent()
    derived.make_dependent_on(
        dependency_expression=f'{repr(1.0 - fraction)} * front + {repr(fraction)} * back',
        dependency_map={'front': front, 'back': back},
    )


def _prepare_gradient_layers(
    front_material: Material,
    back_material: Material,
    discretisation_elements: int,
    interface=None,
) -> LayerCollection:
    """Prepare gradient layers.

    Each sublayer material's SLD/iSLD is constrained to a linear interpolation
    between the front and back material parameters, so that fitting the end
    materials propagates through to the sublayers.
    """
    gradient_sld = _linear_gradient(
        front_value=front_material.sld.value,
        back_value=back_material.sld.value,
        discretisation_elements=discretisation_elements,
    )
    gradient_isld = _linear_gradient(
        front_value=front_material.isld.value,
        back_value=back_material.isld.value,
        discretisation_elements=discretisation_elements,
    )
    gradient_layers = []
    for i in range(discretisation_elements):
        material_i = Material(gradient_sld[i], gradient_isld[i], interface=interface)
        fraction = _gradient_fraction(i, discretisation_elements)
        _constrain_to_endpoints(material_i.sld, front_material.sld, back_material.sld, fraction)
        _constrain_to_endpoints(material_i.isld, front_material.isld, back_material.isld, fraction)
        layer = Layer(
            material=material_i,
            thickness=0.0,
            roughness=0.0,
            name=str(i),
            interface=interface,
            unique_name=global_object.generate_unique_name('GradientLayer'),
        )
        gradient_layers.append(layer)
    return LayerCollection(gradient_layers)
