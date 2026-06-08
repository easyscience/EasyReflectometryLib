# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from types import SimpleNamespace

import pytest
import scipp as sc
from orsopy.fileio import orso

import easyreflectometry
from easyreflectometry.orso_utils import LoadOrso
from easyreflectometry.orso_utils import PolarizedData
from easyreflectometry.orso_utils import _classify_polarization
from easyreflectometry.orso_utils import _get_polarization
from easyreflectometry.orso_utils import _get_sld_values
from easyreflectometry.orso_utils import _normalize_polarization
from easyreflectometry.orso_utils import _spin_from_data_set
from easyreflectometry.orso_utils import load_data_from_orso_file
from easyreflectometry.orso_utils import load_orso_data
from easyreflectometry.orso_utils import load_orso_model
from easyreflectometry.orso_utils import load_polarized_orso_data

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


def _static(name):
    return os.path.join(PATH_STATIC, name)


def _parsed(name):
    """Parse a fixture, suppressing orsopy's ORSOSchemaWarning for raw 'p' codes."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return orso.load_orso(_static(name))


@pytest.fixture
def orso_data():
    """Load the test ORSO data from Ni_example.ort."""
    return orso.load_orso(os.path.join(PATH_STATIC, 'Ni_example.ort'))


def test_load_orso_model(orso_data):
    """Test loading a model from ORSO data."""
    sample = load_orso_model(orso_data)
    assert sample is not None
    assert sample.name == 'Ni on Si'  # Based on the file

    # Verify sample structure: Superphase, Loaded layer, Subphase
    # Stack in file: air | m1 | SiO2 | Si
    assert len(sample) == 3

    # Check Superphase (first layer from stack: air)
    superphase = sample[0]
    assert superphase.name == 'Superphase'
    assert len(superphase.layers) == 1
    assert superphase.layers[0].material.name == 'air'
    assert superphase.layers[0].thickness.value == 0.0
    assert superphase.layers[0].roughness.value == 0.0
    assert superphase.layers[0].thickness.fixed is True
    assert superphase.layers[0].roughness.fixed is True

    # Check Loaded layer (middle layers: m1, SiO2)
    loaded_layer = sample[1]
    assert loaded_layer.name == 'Loaded layer'
    assert len(loaded_layer.layers) == 2
    assert loaded_layer.layers[0].material.name == 'm1'  # Uses original_name, not formula
    assert loaded_layer.layers[0].thickness.value == 1000.0  # From layer definition
    assert loaded_layer.layers[1].material.name == 'SiO2'
    assert loaded_layer.layers[1].thickness.value == 10.0  # From layer definition

    # Check Subphase (last layer from stack: Si)
    subphase = sample[2]
    assert subphase.name == 'Subphase'
    assert len(subphase.layers) == 1
    assert subphase.layers[0].material.name == 'Si'
    assert subphase.layers[0].thickness.value == 0.0
    assert subphase.layers[0].thickness.fixed is True
    # Subphase roughness should be enabled (not fixed)
    assert subphase.layers[0].roughness.fixed is False


def test_load_orso_data(orso_data):
    """Test loading data from ORSO data."""
    data = load_orso_data(orso_data)
    assert data is not None
    # Check structure, e.g., has R_0 in data
    assert 'R_0' in data['data']


def test_LoadOrso(orso_data):
    """Test the LoadOrso function."""
    sample, data = LoadOrso(orso_data)
    assert sample is not None
    assert data is not None
    # Similar checks as above


def test_load_data_from_orso_file():
    """Test loading data from ORSO file."""
    data = load_data_from_orso_file(os.path.join(PATH_STATIC, 'Ni_example.ort'))
    assert data is not None
    # Check it's a sc.DataGroup
    import scipp as sc

    assert isinstance(data, sc.DataGroup)


def test_orso_sld_unit_conversion(orso_data):
    """Test that SLD values from ORSO are correctly converted
       from A^-2 to 10^-6 A^-2.

    ORSO stores SLD in absolute units (A^-2), e.g., 3.47e-06.
    The internal representation uses 10^-6 A^-2,
    so the value should be 3.47.
    """
    sample = load_orso_model(orso_data)

    # Check SiO2 layer (second layer in Loaded layer assembly)
    # ORSO file has: sld: {real: 3.4700000000000002e-06, imag: 0.0}
    # Expected internal value: 3.47
    loaded_layer = sample[1]
    sio2_layer = loaded_layer.layers[1]
    assert sio2_layer.material.name == 'SiO2'
    assert abs(sio2_layer.material.sld.value - 3.47) < 1e-6, (
        f'Expected SLD ~3.47 (10^-6 A^-2), got {sio2_layer.material.sld.value}'
    )

    # Check Si subphase layer
    # ORSO file has: sld: {real: 2.0699999999999997e-06, imag: 0.0}
    # Expected internal value: 2.07
    subphase = sample[2]
    si_layer = subphase.layers[0]
    assert si_layer.material.name == 'Si'
    assert abs(si_layer.material.sld.value - 2.07) < 1e-6, f'Expected SLD ~2.07 (10^-6 A^-2), got {si_layer.material.sld.value}'

    # Check air superphase layer
    # ORSO file has: sld: {real: 0.0, imag: 0.0}
    # Expected internal value: 0.0
    superphase = sample[0]
    air_layer = superphase.layers[0]
    assert air_layer.material.name == 'air'
    assert abs(air_layer.material.sld.value - 0.0) < 1e-6, f'Expected SLD 0.0 (10^-6 A^-2), got {air_layer.material.sld.value}'


def test_LoadOrso_returns_two_items(orso_data):
    """LoadOrso should return exactly two values: (sample, data)."""
    result = LoadOrso(orso_data)
    assert isinstance(result, tuple)
    assert len(result) == 2
    sample, data = result
    assert sample is not None
    assert data is not None


def test_LoadOrso_with_invalid_file(tmp_path):
    """LoadOrso should raise for a corrupt / non-ORSO file."""
    bad_file = tmp_path / 'bad.ort'
    bad_file.write_text('this is not valid ORSO data')
    with pytest.raises((ValueError, Exception)):
        LoadOrso(str(bad_file))


def test_LoadOrso_with_nonexistent_file():
    """LoadOrso should raise for a path that does not exist."""
    with pytest.raises((FileNotFoundError, ValueError, Exception)):
        LoadOrso('/nonexistent/path/to/file.ort')


def test_get_sld_values_defaults_to_zero_when_sld_and_density_missing():
    """_get_sld_values should return (0.0, 0.0) when both
    sld and mass_density are None."""
    material = SimpleNamespace(sld=None, mass_density=None)
    m_sld, m_isld = _get_sld_values(material, 'Unknown')
    assert m_sld == 0.0
    assert m_isld == 0.0


def test_load_orso_model_returns_none_and_warns_when_no_sample_model():
    """load_orso_model should return None and emit a warning
    when the ORSO file has no sample model."""
    orso_data = orso.load_orso(os.path.join(PATH_STATIC, 'test_example1.ort'))
    # Verify the file indeed has no model
    assert orso_data[0].info.data_source.sample.model is None

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = load_orso_model(orso_data)

    assert result is None
    assert len(w) == 1
    assert 'does not contain a sample model definition' in str(w[0].message)


# ---------------------------------------------------------------------------
# Polarization helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'raw, expected',
    [
        (None, None),
        ('unpolarized', 'un'),
        ('un', 'un'),
        ('po', 'po'),
        ('mo', 'mo'),
        ('p', 'p'),
        ('m', 'm'),
        ('pp', 'pp'),
        ('vector', 'vector'),
        ('o', None),  # bare 'o' is unknown, never canonicalized to 'oo'
        ('junk', None),
        (SimpleNamespace(value='po'), 'po'),  # enum-like
    ],
)
def test_normalize_polarization(raw, expected):
    assert _normalize_polarization(raw) == expected


@pytest.mark.parametrize(
    'label, expected',
    [
        ('spin-up', 'up'),
        ('spin_up', 'up'),
        ('spin down', 'down'),
        ('up', 'up'),
        ('down', 'down'),
        ('+', 'up'),
        ('-', 'down'),
        ('spin_three', None),
        (None, None),
    ],
)
def test_spin_from_data_set(label, expected):
    assert _spin_from_data_set(label) == expected


def test_get_polarization_handles_missing_sections():
    # No measurement attribute at all -> None, no exception.
    o = SimpleNamespace(info=SimpleNamespace(data_source=SimpleNamespace()))
    assert _get_polarization(o) is None
    # instrument_settings present but no polarization -> None.
    o = SimpleNamespace(
        info=SimpleNamespace(
            data_source=SimpleNamespace(measurement=SimpleNamespace(instrument_settings=SimpleNamespace(polarization=None)))
        )
    )
    assert _get_polarization(o) is None


# ---------------------------------------------------------------------------
# Classification and polarization-aware loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'fixture, expected',
    [
        ('example.ort', 'unpolarized'),
        ('Ni_example.ort', 'unpolarized'),
        ('NOB_reflectivity_polarized.ort', 'half_polarized'),
        ('test_example2.ort', 'half_polarized'),
        ('test_example3.ort', 'half_polarized'),
    ],
)
def test_classify_polarization(fixture, expected):
    assert _classify_polarization(_parsed(fixture)) == expected


def test_unpolarized_returns_plain_datagroup():
    result = load_polarized_orso_data(_parsed('Ni_example.ort'))
    assert isinstance(result, sc.DataGroup)
    assert not isinstance(result, PolarizedData)


def test_resolved_polarized_enum_codes():
    """NOB: po/mo enum codes, hyphen labels -> resolved up/down."""
    result = load_polarized_orso_data(_parsed('NOB_reflectivity_polarized.ort'))
    assert isinstance(result, PolarizedData)
    assert result.spin_by_key == {'spin-up': 'up', 'spin-down': 'down'}
    assert isinstance(result.raw, sc.DataGroup)
    assert set(result.spin_channels) == {'spin-up', 'spin-down'}


def test_resolved_polarized_raw_p_codes():
    """Raw 'p' files resolve spin from data_set labels (p never implies up)."""
    result = load_polarized_orso_data(_parsed('test_example2.ort'))
    assert isinstance(result, PolarizedData)
    assert result.spin_by_key == {'spin_up': 'up', 'spin_down': 'down'}


def test_unresolved_third_label_falls_back_with_warning():
    """test_example3 has an unrecognized 'spin_three' label -> fallback + warn."""
    with pytest.warns(UserWarning, match='Could not determine the spin direction'):
        result = load_polarized_orso_data(_parsed('test_example3.ort'))
    assert isinstance(result, sc.DataGroup)
    assert not isinstance(result, PolarizedData)
    # All three datasets are preserved in the fallback group.
    assert len(result['data']) == 3


def test_single_channel_half_polarized_warns():
    """test_example1: one spin-up dataset -> PolarizedData + 'no companion' warning."""
    with pytest.warns(UserWarning, match='Only one spin direction'):
        result = load_polarized_orso_data(_parsed('test_example1.ort'))
    assert isinstance(result, PolarizedData)
    assert result.spin_by_key == {'spin_up': 'up'}


def test_contradiction_between_code_and_label_falls_back():
    parsed = _parsed('NOB_reflectivity_polarized.ort')  # po+spin-up, mo+spin-down
    parsed[0].info.data_set = 'spin-down'  # po now claims spin-down -> contradiction
    parsed[1].info.data_set = 'spin-up'
    with pytest.warns(UserWarning, match='contradict'):
        result = load_polarized_orso_data(parsed)
    assert isinstance(result, sc.DataGroup)


def test_missing_data_set_falls_back_no_code_inference():
    """po/mo without data_set must NOT infer spin from the code -> fallback."""
    parsed = _parsed('NOB_reflectivity_polarized.ort')
    parsed[0].info.data_set = None
    parsed[1].info.data_set = None
    with pytest.warns(UserWarning, match='Could not determine'):
        result = load_polarized_orso_data(parsed)
    assert isinstance(result, sc.DataGroup)


def test_unsupported_state_falls_back_preserving_all_datasets():
    parsed = _parsed('NOB_reflectivity_polarized.ort')
    parsed[0].info.data_source.measurement.instrument_settings.polarization = 'pp'
    parsed[1].info.data_source.measurement.instrument_settings.polarization = 'mm'
    with pytest.warns(UserWarning, match='not yet supported'):
        result = load_polarized_orso_data(parsed)
    assert isinstance(result, sc.DataGroup)
    assert len(result['data']) == 2


def test_mixed_un_and_polarized_is_unsupported():
    parsed = _parsed('NOB_reflectivity_polarized.ort')
    parsed[0].info.data_source.measurement.instrument_settings.polarization = 'unpolarized'
    assert _classify_polarization(parsed) == 'unsupported'
    with pytest.warns(UserWarning, match='unsupported mix'):
        result = load_polarized_orso_data(parsed)
    assert isinstance(result, sc.DataGroup)


def test_load_orso_data_disambiguates_duplicate_data_set_keys():
    """Repeated data_set labels must not overwrite each other (review #5)."""
    parsed = _parsed('NOB_reflectivity_polarized.ort')
    parsed[0].info.data_set = 'spin-up'
    parsed[1].info.data_set = 'spin-up'
    data_group = load_orso_data(parsed)
    assert set(data_group['data']) == {'R_spin-up', 'R_spin-up_1'}
    assert set(data_group['coords']) == {'Qz_spin-up', 'Qz_spin-up_1'}


def test_load_orso_data_tags_spin_and_polarization_in_attrs():
    data_group = load_orso_data(_parsed('NOB_reflectivity_polarized.ort'))
    up = data_group['attrs']['R_spin-up']
    assert up['polarization'].value == 'half_polarized'
    assert up['spin'].value == 'up'
    assert data_group['attrs']['R_spin-down']['spin'].value == 'down'


def test_loadorso_return_type_is_stable_for_polarized_file():
    """LoadOrso must still return (Sample-or-None, sc.DataGroup), never PolarizedData."""
    sample, data = LoadOrso(_parsed('NOB_reflectivity_polarized.ort'))
    assert isinstance(data, sc.DataGroup)
    assert not isinstance(data, PolarizedData)
