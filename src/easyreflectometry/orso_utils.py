import logging
import warnings

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


def LoadOrso(orso_data):
    """Load a model from an ORSO file."""

    orso_obj = _coerce_orso_object(orso_data)
    sample = load_orso_model(orso_obj)
    data = load_orso_data(orso_obj)
    return sample, data


def _coerce_orso_object(orso_input):
    """Return a parsed ORSO object list from either a path or pre-parsed input."""
    try:
        if orso_input and hasattr(orso_input[0], 'info'):
            return orso_input
    except (TypeError, IndexError):
        pass
    return orso.load_orso(orso_input)


def load_data_from_orso_file(fname: str) -> sc.DataGroup:
    """Load data from an ORSO file."""
    try:
        orso_data = orso.load_orso(fname)
    except Exception as e:
        raise ValueError(f'Error loading ORSO file: {e}')
    return load_orso_data(orso_data)


def load_orso_model(orso_data) -> Sample:
    """
    Load a model from an ORSO file and return a Sample object.

    The ORSO file .ort contains information about the sample, saved
    as a simple "stack" string, e.g. 'air | m1 | SiO2 | Si'.
    This gets parsed by the ORSO library and converted into an ORSO Dataset object.

    The stack is converted to a proper Sample structure:
    - First layer -> Superphase assembly (thickness=0, roughness=0, both fixed)
    - Middle layers -> 'Loaded layer' Multilayer assembly (parameters enabled)
    - Last layer -> Subphase assembly (thickness=0 fixed, roughness enabled)

    :param orso_data: Parsed ORSO dataset list (as returned by ``orso.load_orso``).
    :type orso_data: list
    :return: An EasyReflectometry Sample object.
    :rtype: Sample
    :raises ValueError: If ORSO layers could not be resolved or fewer than 2 layers.
    """
    # Extract stack string and layer definitions from ORSO sample model
    sample_model = orso_data[0].info.data_source.sample.model
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
    sample_info = orso_data[0].info.data_source.sample
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
    """Extract SLD values from material, calculating from density if needed

    Note: ORSO stores SLD in absolute units (A^-2), but the internal representation
    uses 10^-6 A^-2. When reading directly from ORSO, we multiply by 1e6 to convert.
    When calculating from mass density, MaterialDensity already returns the correct units.
    """
    if material.sld is None and material.mass_density is not None:
        # Calculate SLD from mass density
        # MaterialDensity already returns values in 10^-6 A^-2 units
        m_density = material.mass_density.magnitude
        density = MaterialDensity(chemical_structure=material_name, density=m_density)
        m_sld = density.sld.value
        m_isld = density.isld.value
    elif material.sld is None:
        # No SLD and no mass density available, default to 0.0
        m_sld = 0.0
        m_isld = 0.0
    else:
        # ORSO stores SLD in absolute units (A^-2)
        # Convert to internal representation (10^-6 A^-2) by multiplying by 1e6
        if isinstance(material.sld, ComplexValue):
            raw_sld = material.sld.real
            m_sld = raw_sld * 1e6
            m_isld = material.sld.imag * 1e6
        else:
            raw_sld = material.sld
            m_sld = raw_sld * 1e6
            m_isld = 0.0
        if raw_sld != 0.0 and abs(raw_sld) > 1e-2:
            warnings.warn(
                f'ORSO SLD value {raw_sld} for "{material_name}" seems large for '
                f'absolute units (A^-2). Verify the file stores SLD in A^-2, not '
                f'10^-6 A^-2, as the value is multiplied by 1e6 internally.',
                UserWarning,
                stacklevel=3,
            )

    return m_sld, m_isld


def load_orso_data(orso_data) -> DataSet1D:
    """Convert parsed ORSO dataset objects into a scipp DataGroup.

    :param orso_data: Parsed ORSO dataset list (as returned by ``orso.load_orso``).
    :type orso_data: list
    :return: A scipp DataGroup with data, coords, and attrs.
    :rtype: sc.DataGroup
    """
    data = {}
    coords = {}
    attrs = {}
    for i, o in enumerate(orso_data):
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
