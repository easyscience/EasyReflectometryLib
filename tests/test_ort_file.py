# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 DMSC

import logging

import numpy as np

# from dmsc_nightly.data import make_pooch
import pooch
import pytest
from easyscience.fitting import AvailableMinimizers

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import load
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample


def make_pooch(base_url: str, registry: dict[str, str | None]) -> pooch.Pooch:
    """Make a Pooch object to download test data."""
    return pooch.create(
        path=pooch.os_cache("data"),
        env="POOCH_DIR",
        base_url=base_url,
        registry=registry,
    )


@pytest.fixture(scope="module")
def data_registry():
    return make_pooch(
        base_url="https://pub-6c25ef91903d4301a3338bd53b370098.r2.dev",
        registry={
            "amor_reduced_iofq.ort": None,
        },
    )


@pytest.fixture(scope="module")
def load_data(data_registry):
    path = data_registry.fetch("amor_reduced_iofq.ort")
    logging.info("Loading data from %s", path)
    data = load(path)
    return data


@pytest.fixture(scope="module")
def fit_model(load_data):
    data = load_data
    # Rescale data
    reflectivity = data["data"]["R_0"].values
    scale_factor = 1 / np.max(reflectivity)
    data["data"]["R_0"].values *= scale_factor

    # Create a model for the sample

    si = Material(sld=2.07, isld=0.0, name="Si")
    sio2 = Material(sld=3.47, isld=0.0, name="SiO2")
    d2o = Material(sld=6.33, isld=0.0, name="D2O")
    dlipids = Material(sld=5.0, isld=0.0, name="DLipids")

    superphase = Layer(material=si, thickness=0, roughness=0, name="Si superphase")
    sio2_layer = Layer(material=sio2, thickness=20, roughness=4, name="SiO2 layer")
    dlipids_layer = Layer(
        material=dlipids, thickness=40, roughness=4, name="DLipids layer"
    )
    subphase = Layer(material=d2o, thickness=0, roughness=5, name="D2O subphase")

    multi_sample = Sample(
        Multilayer(superphase),
        Multilayer(sio2_layer),
        Multilayer(dlipids_layer),
        Multilayer(subphase),
        name="Multilayer Structure",
    )

    multi_layer_model = Model(
        sample=multi_sample,
        scale=1,
        background=0.000001,
        resolution_function=PercentageFwhm(0),
        name="Multilayer Model",
    )

    # Set the fitting parameters

    sio2_layer.roughness.bounds = (3, 12)
    sio2_layer.material.sld.bounds = (3.47, 5)
    sio2_layer.thickness.bounds = (10, 30)

    subphase.material.sld.bounds = (6, 6.35)
    dlipids_layer.thickness.bounds = (30, 60)
    dlipids_layer.roughness.bounds = (3, 10)
    dlipids_layer.material.sld.bounds = (4, 6)
    multi_layer_model.scale.bounds = (0.8, 1.2)
    multi_layer_model.background.bounds = (1e-6, 1e-3)

    sio2_layer.roughness.free = True
    sio2_layer.material.sld.free = True
    sio2_layer.thickness.free = True
    subphase.material.sld.free = True
    dlipids_layer.thickness.free = True
    dlipids_layer.roughness.free = True
    dlipids_layer.material.sld.free = True
    multi_layer_model.scale.free = True
    multi_layer_model.background.free = True

    # Run the model and plot the results

    multi_layer_model.interface = CalculatorFactory()

    fitter1 = MultiFitter(multi_layer_model)
    fitter1.switch_minimizer(AvailableMinimizers.Bumps_simplex)

    analysed = fitter1.fit(data)
    return analysed


def test_read_reduced_data__check_structure(load_data):
    data_keys = load_data["data"].keys()
    coord_keys = load_data["coords"].keys()
    for key in data_keys:
        if key in coord_keys:
            assert len(load_data["data"][key].values) == len(
                load_data["coords"][key].values
            )


def test_validate_physical_data__r_values_non_negative(load_data):
    for key in load_data["data"].keys():
        assert all(load_data["data"][key].values >= 0)


def test_validate_physical_data__r_values_finite(load_data):
    for key in load_data["data"].keys():
        assert all(np.isfinite(load_data["data"][key].values))


@pytest.mark.skip("Currently no warning implemented")
def test_validate_physical_data__r_values_ureal_positive(load_data):
    a = load_data["data"]["R_0"].values
    b = 1 + 2 * np.sqrt(load_data["data"]["R_0"].variances)
    for val_a, val_b in zip(a, b):
        if val_a > val_b:
            pytest.warns(
                UserWarning,
                reason=f"Reflectivity value {val_a} is unphysically large compared to its uncertainty {val_b}"
            )
    assert all(
        load_data["data"]["R_0"].values
        <= 1 + 2 * np.sqrt(load_data["data"]["R_0"].variances)
    )


def test_validate_physical_data__q_values_non_negative(load_data):
    for key in load_data["coords"].keys():
        assert all(load_data["coords"][key].values >= 0)


def test_validate_physical_data__q_values_ureal_positive(load_data):
    for key in load_data["coords"].keys():
        # Reflectometry data is usually with the range of 0-5,
        # so 10 is a safe upper limit
        assert all(load_data["coords"][key].values < 10)


def test_validate_physical_data__q_values_finite(load_data):
    for key in load_data["coords"].keys():
        assert all(np.isfinite(load_data["coords"][key].values < 10))


@pytest.mark.skip("Currently no meta data to check")
def test_validate_meta_data__required_meta_data() -> None:
    pytest.fail(reason="Currently no meta data to check")


def test_analyze_reduced_data__fit_model_success(fit_model):
    assert fit_model["success"] is True


def test_analyze_reduced_data__fit_model_reasonable(fit_model):
    assert fit_model["reduced_chi"] < 0.01
