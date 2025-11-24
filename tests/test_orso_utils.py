# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 DMSC

import pytest
from orsopy.fileio import orso

from easyreflectometry.orso_utils import (
    LoadOrso,
    load_data_from_orso_file,
    load_orso_data,
    load_orso_model,
)


@pytest.fixture
def orso_data():
    """Load the test ORSO data from Ni_example.ort."""
    return orso.load_orso("Ni_example.ort")


def test_load_orso_model(orso_data):
    """Test loading a model from ORSO data."""
    sample = load_orso_model(orso_data)
    assert sample is not None
    assert sample.name == "Ni on Si"  # Based on the file


def test_load_orso_data(orso_data):
    """Test loading data from ORSO data."""
    data = load_orso_data(orso_data)
    assert data is not None
    # Check structure, e.g., has R_0 in data
    assert "R_0" in data["data"]


def test_LoadOrso(orso_data):
    """Test the LoadOrso function."""
    sample, data = LoadOrso(orso_data)
    assert sample is not None
    assert data is not None
    # Similar checks as above


def test_load_data_from_orso_file():
    """Test loading data from ORSO file."""
    data = load_data_from_orso_file("Ni_example.ort")
    assert data is not None
    # Check it's a sc.DataGroup
    import scipp as sc
    assert isinstance(data, sc.DataGroup)