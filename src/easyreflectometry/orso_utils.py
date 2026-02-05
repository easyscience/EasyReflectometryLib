import logging

import numpy as np
import scipp as sc
from orsopy.fileio import Header
from orsopy.fileio import model_language
from orsopy.fileio import orso
from orsopy.fileio.base import ComplexValue

from easyreflectometry.data import DataSet1D

from .sample.assemblies.multilayer import Multilayer
from .sample.collections.sample import Sample
from .sample.elements.layers.layer import Layer
from .sample.elements.materials.material import Material
from .sample.elements.materials.material_density import MaterialDensity

# Set up logging
logger = logging.getLogger(__name__)


def LoadOrso(orso_str: str):
    """Load a model from an ORSO file."""

    sample = load_orso_model(orso_str)
    data = load_orso_data(orso_str)

    return sample, data


def load_data_from_orso_file(fname: str) -> sc.DataGroup:
    """Load data from an ORSO file."""
    try:
        orso_data = orso.load_orso(fname)
    except Exception as e:
        raise ValueError(f'Error loading ORSO file: {e}')
    return load_orso_data(orso_data)


def load_orso_model(orso_str: str) -> Sample:
    """
    Load a model from an ORSO file and return a Sample object.

    The ORSO file .ort contains information about the sample, saved
    as a simple "stack" string, e.g. 'air | m1 | SiO2 | Si'.
    This gets parsed by the ORSO library and converted into an ORSO Dataset object.

    The stack is converted to a proper Sample structure:
    - First layer -> Superphase assembly (thickness=0, roughness=0, both fixed)
    - Middle layers -> 'Loaded layer' Multilayer assembly (parameters enabled)
    - Last layer -> Subphase assembly (thickness=0 fixed, roughness enabled)

    Args:
        orso_str: The ORSO file content as a string

    Returns:
        Sample: An EasyReflectometry Sample object

    Raises:
        ValueError: If ORSO layers could not be resolved or fewer than 2 layers
    """
    # Extract stack string and layer definitions from ORSO sample model
    sample_model = orso_str[0].info.data_source.sample.model
    stack_str = sample_model.stack
    layers_dict = sample_model.layers if hasattr(sample_model, 'layers') else None
    orso_sample = model_language.SampleModel(stack=stack_str, layers=layers_dict)

    # Try to resolve layers using different methods
    try:
        orso_layers = orso_sample.resolve_to_layers()
    except ValueError:
        orso_layers = orso_sample.resolve_stack()

    # Handle case where layers are not resolved correctly
    if not orso_layers:
        raise ValueError('Could not resolve ORSO layers.')

    if len(orso_layers) < 2:
        raise ValueError('ORSO stack must contain at least 2 layers (superphase and subphase).')

    logger.debug(f'Resolved layers: {orso_layers}')

    # Convert ORSO layers to EasyReflectometry layers
    erl_layers = []
    for layer in orso_layers:
        erl_layer = _convert_orso_layer_to_erl(layer)
        erl_layers.append(erl_layer)

    # Create Superphase from first layer (thickness=0, roughness=0, both fixed)
    superphase_layer = erl_layers[0]
    superphase_layer.thickness.value = 0.0
    superphase_layer.roughness.value = 0.0
    superphase_layer.thickness.fixed = True
    superphase_layer.roughness.fixed = True
    superphase = Multilayer(superphase_layer, name='Superphase')

    # Create Subphase from last layer (thickness=0 fixed, roughness enabled)
    subphase_layer = erl_layers[-1]
    subphase_layer.thickness.value = 0.0
    subphase_layer.thickness.fixed = True
    subphase_layer.roughness.fixed = False
    subphase = Multilayer(subphase_layer, name='Subphase')

    # Create Sample from the file
    sample_info = orso_str[0].info.data_source.sample
    sample_name = sample_info.name if sample_info.name else 'ORSO Sample'

    # Build Sample based on number of layers
    if len(erl_layers) == 2:
        # Only superphase and subphase, no middle layers
        sample = Sample(superphase, subphase, name=sample_name)
    else:
        # Create middle layer assembly from layers between first and last
        middle_layers = erl_layers[1:-1]
        loaded_layer = Multilayer(middle_layers, name='Loaded layer')
        sample = Sample(superphase, loaded_layer, subphase, name=sample_name)

    return sample


def _convert_orso_layer_to_erl(layer):
    """Helper function to convert an ORSO layer to an EasyReflectometry layer"""
    material = layer.material
    # Prefer original_name for material name, fall back to formula if available
    m_name = layer.original_name if layer.original_name is not None else material.formula

    # Get SLD values (use formula for density calculation if available)
    formula_for_calc = material.formula if material.formula is not None else m_name
    m_sld, m_isld = _get_sld_values(material, formula_for_calc)

    # Create and return ERL layer
    return Layer(
        material=Material(sld=m_sld, isld=m_isld, name=m_name),
        thickness=layer.thickness.magnitude if layer.thickness is not None else 0.0,
        roughness=layer.roughness.magnitude if layer.roughness is not None else 0.0,
        name=layer.original_name if layer.original_name is not None else m_name,
    )


def _get_sld_values(material, material_name):
    """Extract SLD values from material, calculating from density if needed"""
    if material.sld is None and material.mass_density is not None:
        # Calculate SLD from mass density
        m_density = material.mass_density.magnitude
        density = MaterialDensity(chemical_structure=material_name, density=m_density)
        m_sld = density.sld.value
        m_isld = density.isld.value
    else:
        if isinstance(material.sld, ComplexValue):
            m_sld = material.sld.real
            m_isld = material.sld.imag
        else:
            m_sld = material.sld
            m_isld = 0.0

    return m_sld, m_isld


def load_orso_data(orso_str: str) -> DataSet1D:
    data = {}
    coords = {}
    attrs = {}
    for i, o in enumerate(orso_str):
        name = i
        if o.info.data_set is not None:
            name = o.info.data_set
        coords[f'Qz_{name}'] = sc.array(
            dims=[f'{o.info.columns[0].name}_{name}'],
            values=o.data[:, 0],
            variances=np.square(o.data[:, 3]),
            unit=sc.Unit(o.info.columns[0].unit),
        )
        try:
            data[f'R_{name}'] = sc.array(
                dims=[f'{o.info.columns[0].name}_{name}'],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
                unit=sc.Unit(o.info.columns[1].unit),
            )
        except TypeError:
            data[f'R_{name}'] = sc.array(
                dims=[f'{o.info.columns[0].name}_{name}'],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
            )
        attrs[f'R_{name}'] = {'orso_header': sc.scalar(Header.asdict(o.info))}
    data_group = sc.DataGroup(data=data, coords=coords, attrs=attrs)
    return data_group
