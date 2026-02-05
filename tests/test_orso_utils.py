# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 DMSC

import os

import pytest
from orsopy.fileio import orso

import easyreflectometry
from easyreflectometry.orso_utils import LoadOrso
from easyreflectometry.orso_utils import load_data_from_orso_file
from easyreflectometry.orso_utils import load_orso_data
from easyreflectometry.orso_utils import load_orso_model

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


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
