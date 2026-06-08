# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

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
    """Load a model from an ORSO file and return a Sample object.

    The ORSO file .ort contains information about the sample, saved
    as a simple "stack" string, e.g. 'air | m1 | SiO2 | Si'.
    This gets parsed by the ORSO library and converted into an ORSO Dataset object.

    The stack is converted to a proper Sample structure:
    - First layer -> Superphase assembly (thickness=0, roughness=0, both fixed)
    - Middle layers -> 'Loaded layer' Multilayer assembly (parameters enabled)
    - Last layer -> Subphase assembly (thickness=0 fixed, roughness enabled)

    Parameters
    ----------
    orso_data : list
        Parsed ORSO dataset list (as returned by ``orso.load_orso``).

    Raises
    ------
    ValueError :
        If ORSO layers could not be resolved or fewer than 2 layers.

    Returns
    -------
    Sample
        An EasyReflectometry Sample object.
    """
    # Extract stack string and layer definitions from ORSO sample model
    sample_model = orso_data[0].info.data_source.sample.model
    if sample_model is None:
        warnings.warn(
            'ORSO file does not contain a sample model definition. Only experimental data can be loaded from this file.',
            UserWarning,
            stacklevel=2,
        )
        return None
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
    r"""Helper function to convert an ORSO layer to an EasyReflectometry laye."""
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
    When calculating from mass density, MaterialDensity already returns the correct units..
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

    Parameters
    ----------
    orso_data : list
        Parsed ORSO dataset list (as returned by ``orso.load_orso``).

    Returns
    -------
    sc.DataGroup
        A scipp DataGroup with data, coords, and attrs.
    """
    data = {}
    coords = {}
    attrs = {}
    # Tag the group with polarization classification and, when the file is a
    # resolved half-polarized measurement, the per-dataset spin direction. These
    # are in-band markers so consumers can read spin without the typed container.
    classification = _classify_polarization(orso_data)
    spins = _resolve_spins(orso_data) if classification == 'half_polarized' else None
    used_names: dict = {}
    for i, o in enumerate(orso_data):
        base = o.info.data_set if o.info.data_set is not None else i
        # Disambiguate repeated data_set labels so no dataset is silently
        # overwritten (first use: <base>, repeats: <base>_1, <base>_2, ...).
        name = _unique_name(base, used_names)
        dim = f'{o.info.columns[0].name}_{name}'
        coords[f'Qz_{name}'] = sc.array(
            dims=[dim],
            values=o.data[:, 0],
            variances=np.square(o.data[:, 3]),
            unit=sc.Unit(o.info.columns[0].unit),
        )
        try:
            data[f'R_{name}'] = sc.array(
                dims=[dim],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
                unit=sc.Unit(o.info.columns[1].unit),
            )
        except TypeError:
            data[f'R_{name}'] = sc.array(
                dims=[dim],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
            )
        dataset_attrs = {
            'orso_header': sc.scalar(Header.asdict(o.info)),
            'polarization': sc.scalar(classification),
        }
        if spins is not None:
            dataset_attrs['spin'] = sc.scalar(spins[i])
        attrs[f'R_{name}'] = dataset_attrs
    data_group = sc.DataGroup(data=data, coords=coords, attrs=attrs)
    return data_group


def _unique_name(base, used_names: dict) -> str:
    """Return a unique string key for *base*, suffixing repeats with _1, _2, ..."""
    key = str(base)
    count = used_names.get(key, 0)
    used_names[key] = count + 1
    return key if count == 0 else f'{key}_{count}'


# ---------------------------------------------------------------------------
# Polarization handling
# ---------------------------------------------------------------------------

# ORSO-allowed polarization states. ``p`` / ``m`` are not in this list but appear
# in real files as a legacy polarized-presence hint (they carry no spin meaning).
_UNPOLARIZED = {'un'}
_HALF_POLARIZED = {'po', 'mo', 'p', 'm'}


@dataclass
class PolarizedData:
    """Typed view of a resolved half-polarized ORSO load.

    Attributes
    ----------
    polarization:
        Canonical classification, e.g. ``'half_polarized'``.
    spin_channels:
        Mapping of dataset label -> single-dataset ``sc.DataGroup``.
    spin_by_key:
        Mapping of dataset label -> ``'up'`` / ``'down'``.
    raw:
        The full flat ``sc.DataGroup`` (all channels), for consumers that only
        understand the legacy structure.
    """

    polarization: str
    spin_channels: dict
    spin_by_key: dict
    raw: sc.DataGroup


def _get_polarization(o) -> Optional[str]:
    """Per-dataset polarization value as a raw string, or None if absent.

    Tolerant of a missing ``measurement`` / ``instrument_settings`` section and
    of the value being an orsopy enum (``.value``) or a raw string.
    """
    try:
        raw = o.info.data_source.measurement.instrument_settings.polarization
    except AttributeError:
        return None
    if raw is None:
        return None
    return raw.value if hasattr(raw, 'value') else str(raw)


def _normalize_polarization(raw) -> Optional[str]:
    """Normalize a polarization value to a canonical code, or None if unknown.

    Allowed vocabulary: ``un po mo op om pp pm mp mm vector``. Single-letter
    ``p`` / ``m`` are kept as legacy presence hints (never expanded to po/mo).
    A bare ``o`` (and anything else) is unknown -> None (no invented ``oo``).
    """
    if raw is None:
        return None
    if hasattr(raw, 'value'):
        raw = raw.value
    code = str(raw).strip().lower()
    if code.startswith('un'):
        return 'un'
    if code == 'vector':
        return 'vector'
    if code in {'po', 'mo', 'op', 'om', 'pp', 'pm', 'mp', 'mm'}:
        return code
    if code in {'p', 'm'}:
        return code
    return None


def _classify_polarization(orso_data) -> str:
    """Classify a file as 'unpolarized', 'half_polarized', or 'unsupported'."""
    codes = []
    for o in orso_data:
        norm = _normalize_polarization(_get_polarization(o))
        # Absent / unknown metadata counts as unpolarized for classification.
        codes.append(norm if norm is not None else 'un')
    if all(c in _UNPOLARIZED for c in codes):
        return 'unpolarized'
    if all(c in _HALF_POLARIZED for c in codes):
        return 'half_polarized'
    return 'unsupported'


def _spin_from_data_set(label) -> Optional[str]:
    """Map a ``data_set`` label to 'up' / 'down', or None if unrecognized."""
    if label is None:
        return None
    text = str(label).strip().lower()
    # Bare sign characters must be matched before separator normalization, which
    # would otherwise consume the '-' used as a hyphen in e.g. 'spin-down'.
    if text == '+':
        return 'up'
    if text == '-':
        return 'down'
    for sep in ('-', '_'):
        text = text.replace(sep, ' ')
    text = ' '.join(text.split())
    if text.startswith('spin '):
        text = text[len('spin ') :]
    if text in {'up', 'u', 'plus'}:
        return 'up'
    if text in {'down', 'd', 'minus'}:
        return 'down'
    return None


def _spin_from_polarization_code(code) -> Optional[str]:
    """Expected spin implied by a polarization code (cross-check only).

    Only genuine two-letter incident codes carry spin meaning. Legacy ``p`` /
    ``m`` and everything else return None (no spin meaning).
    """
    if code in {'po', 'pp', 'pm'}:
        return 'up'
    if code in {'mo', 'mp', 'mm'}:
        return 'down'
    return None


def _resolve_spins_with_reason(orso_data):
    """Resolve a spin per dataset from ``data_set`` metadata.

    Returns ``(spins, None)`` when every dataset has a recognized, unique
    ``data_set`` spin label that does not contradict its polarization code;
    otherwise ``(None, reason)``.
    """
    spins = []
    seen = set()
    for o in orso_data:
        label = o.info.data_set
        spin = _spin_from_data_set(label)
        if spin is None:
            return None, f'unrecognized data_set label {label!r}'
        identity = str(label).strip().lower()
        if identity in seen:
            return None, f'duplicate data_set label {label!r}'
        seen.add(identity)
        expected = _spin_from_polarization_code(_normalize_polarization(_get_polarization(o)))
        if expected is not None and expected != spin:
            return None, f'polarization code contradicts data_set spin {spin!r}'
        spins.append(spin)
    return spins, None


def _resolve_spins(orso_data) -> Optional[list]:
    """Spin per dataset, or None if the assignment is not unequivocal."""
    spins, _reason = _resolve_spins_with_reason(orso_data)
    return spins


def load_polarized_orso_data(orso_data):
    """Polarization-aware load.

    Returns a :class:`PolarizedData` for a resolved half-polarized file, and a
    plain ``sc.DataGroup`` otherwise (unpolarized, unsupported, or unresolved
    fallback). Emits a ``UserWarning`` whenever it falls back or when only a
    single spin channel is present.
    """
    orso_obj = _coerce_orso_object(orso_data)
    classification = _classify_polarization(orso_obj)

    if classification == 'unpolarized':
        return load_orso_data(orso_obj)

    if classification == 'unsupported':
        codes_present = sorted({_normalize_polarization(_get_polarization(o)) or 'unknown' for o in orso_obj})
        unsupported_codes = [c for c in codes_present if c not in (_UNPOLARIZED | _HALF_POLARIZED)]
        if unsupported_codes:
            detail = f'state(s) {unsupported_codes} not yet supported'
        else:
            # Every code is individually supported, but the file mixes classes
            # (e.g. unpolarized + polarized) which we cannot resolve as one set.
            detail = f'unsupported mix of states {codes_present}'
        warnings.warn(
            f'ORSO polarization {detail}; loading all datasets without spin assignment.',
            UserWarning,
            stacklevel=2,
        )
        return load_orso_data(orso_obj)

    # half_polarized
    spins, reason = _resolve_spins_with_reason(orso_obj)
    if spins is None:
        warnings.warn(
            f'Could not determine the spin direction for all datasets ({reason}); '
            'loading as a standard multi-dataset experiment.',
            UserWarning,
            stacklevel=2,
        )
        return load_orso_data(orso_obj)

    spin_channels = {}
    spin_by_key = {}
    for o, spin in zip(orso_obj, spins):
        key = str(o.info.data_set)
        spin_channels[key] = load_orso_data([o])
        spin_by_key[key] = spin

    if len(orso_obj) == 1:
        warnings.warn(
            f'Only one spin direction ({spins[0]!r}) present; no companion channel.',
            UserWarning,
            stacklevel=2,
        )

    return PolarizedData(
        polarization=classification,
        spin_channels=spin_channels,
        spin_by_key=spin_by_key,
        raw=load_orso_data(orso_obj),
    )
