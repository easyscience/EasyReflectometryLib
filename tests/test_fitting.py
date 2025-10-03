__author__ = 'github.com/arm61'

import os

import pytest
from easyscience.fitting.minimizers.factory import AvailableMinimizers

import easyreflectometry
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data.measurement import load
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


@pytest.mark.parametrize('minimizer', [AvailableMinimizers.Bumps, AvailableMinimizers.LMFit])
def test_fitting(minimizer):
    fpath = os.path.join(PATH_STATIC, 'example.ort')
    data = load(fpath)
    si = Material(2.07, 0, 'Si')
    sio2 = Material(3.47, 0, 'SiO2')
    film = Material(2.0, 0, 'Film')
    d2o = Material(6.36, 0, 'D2O')
    si_layer = Layer(si, 0, 0, 'Si layer')
    sio2_layer = Layer(sio2, 30, 3, 'SiO2 layer')
    film_layer = Layer(film, 250, 3, 'Film Layer')
    superphase = Layer(d2o, 0, 3, 'D2O Subphase')
    sample = Sample(
        Multilayer(si_layer),
        Multilayer(sio2_layer),
        Multilayer(film_layer),
        Multilayer(superphase),
        name='Film Structure',
    )
    resolution_function = PercentageFwhm(0.02)
    model = Model(sample, 1, 1e-6, resolution_function, 'Film Model')
    # Thicknesses
    sio2_layer.thickness.fixed = False
    sio2_layer.thickness.bounds = (15, 50)
    film_layer.thickness.fixed = False
    film_layer.thickness.bounds = (200, 300)
    # Roughnesses
    si_layer.roughness.fixed = True
    sio2_layer.roughness.bounds = (1, 15)
    film_layer.roughness.fixed = False
    film_layer.roughness.bounds = (1, 15)
    superphase.roughness.fixed = True
    superphase.roughness.bounds = (1, 15)
    # Scattering length density
    film.sld.fixed = False
    film.sld.bounds = (0.1, 3)
    # Background
    model.background.fixed = False
    model.background.bounds = (1e-7, 1e-5)
    # Scale
    model.scale.fixed = False
    model.scale.bounds = (0.5, 1.5)
    interface = CalculatorFactory()
    model.interface = interface
    fitter = MultiFitter(model)
    fitter.easy_science_multi_fitter.switch_minimizer(minimizer)
    analysed = fitter.fit(data)
    assert 'R_0_model' in analysed.keys()
    assert 'SLD_0' in analysed.keys()
    assert 'success' in analysed.keys()
    assert analysed['success']


def test_fitting_with_zero_variance():
    """Test that zero variance points are properly detected and masked during fitting when present in the data."""
    import warnings

    import numpy as np

    from easyreflectometry.data.measurement import load

    # Load data that contains zero variance points
    fpath = os.path.join(PATH_STATIC, 'ref_zero_var.txt')

    # First, load the raw data to count zero variance points
    raw_data = np.loadtxt(fpath, delimiter=',', comments='#')
    zero_variance_count = np.sum(raw_data[:, 2] == 0.0)  # Error column
    assert zero_variance_count == 6, f"Expected 6 zero variance points, got {zero_variance_count}"

    # Load data through the measurement module (which already filters zero variance)
    data = load(fpath)

    # Create a simple model for fitting
    si = Material(2.07, 0, 'Si')
    sio2 = Material(3.47, 0, 'SiO2')
    film = Material(2.0, 0, 'Film')
    d2o = Material(6.36, 0, 'D2O')
    si_layer = Layer(si, 0, 0, 'Si layer')
    sio2_layer = Layer(sio2, 30, 3, 'SiO2 layer')
    film_layer = Layer(film, 250, 3, 'Film Layer')
    superphase = Layer(d2o, 0, 3, 'D2O Subphase')
    sample = Sample(
        Multilayer(si_layer),
        Multilayer(sio2_layer),
        Multilayer(film_layer),
        Multilayer(superphase),
        name='Film Structure',
    )
    resolution_function = PercentageFwhm(0.02)
    model = Model(sample, 1, 1e-6, resolution_function, 'Film Model')

    # Set some parameters as fittable
    sio2_layer.thickness.fixed = False
    sio2_layer.thickness.bounds = (15, 50)
    film_layer.thickness.fixed = False
    film_layer.thickness.bounds = (200, 300)
    film.sld.fixed = False
    film.sld.bounds = (0.1, 3)
    model.background.fixed = False
    model.background.bounds = (1e-7, 1e-5)
    model.scale.fixed = False
    model.scale.bounds = (0.5, 1.5)

    interface = CalculatorFactory()
    model.interface = interface
    fitter = MultiFitter(model)

    # Capture warnings during fitting - check if zero variance points still exist in the data
    # and are properly handled by the fitting method
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        analysed = fitter.fit(data)

        # Check if any zero variance warnings were issued during fitting
        fitting_warnings = [str(warning.message) for warning in w 
                          if "zero variance during fitting" in str(warning.message)]

        # The fitting method should handle zero variance points gracefully
        # If there are any zero variance points remaining in the data, they should be masked
        # and a warning should be issued
        if len(fitting_warnings) > 0:
            # Verify the warning message format and that it mentions masking points
            for warning_msg in fitting_warnings:
                assert "Masked" in warning_msg and "zero variance during fitting" in warning_msg
                print(f"Info: {warning_msg}")  # Log for debugging

    # Basic checks that fitting completed
    # The keys will be based on the filename, not just '0'
    model_keys = [k for k in analysed.keys() if k.endswith('_model')]
    sld_keys = [k for k in analysed.keys() if k.startswith('SLD_')]
    assert len(model_keys) > 0, f"No model keys found in {list(analysed.keys())}"
    assert len(sld_keys) > 0, f"No SLD keys found in {list(analysed.keys())}"
    assert 'success' in analysed.keys()


def test_fitting_with_manual_zero_variance():
    """Test the fit method with manually created zero variance points."""
    import warnings

    import numpy as np
    import scipp as sc

    # Create synthetic data with some zero variance points
    qz_values = np.linspace(0.01, 0.3, 50)
    r_values = np.exp(-qz_values * 100) + 0.01  # Synthetic reflectivity data

    # Create variances with some zero values
    variances = np.ones_like(r_values) * 0.01**2
    # Set some variances to zero (simulate bad data points)
    variances[10:15] = 0.0  # 5 zero variance points
    variances[30:32] = 0.0  # 2 more zero variance points

    # Create scipp DataGroup manually
    data = sc.DataGroup({
        'coords': {
            'Qz_0': sc.array(dims=['Qz_0'], values=qz_values)
        },
        'data': {
            'R_0': sc.array(dims=['Qz_0'], values=r_values, variances=variances)
        }
    })

    # Create a simple model for fitting
    si = Material(2.07, 0, 'Si')
    sio2 = Material(3.47, 0, 'SiO2')
    film = Material(2.0, 0, 'Film')
    d2o = Material(6.36, 0, 'D2O')
    si_layer = Layer(si, 0, 0, 'Si layer')
    sio2_layer = Layer(sio2, 30, 3, 'SiO2 layer')
    film_layer = Layer(film, 250, 3, 'Film Layer')
    superphase = Layer(d2o, 0, 3, 'D2O Subphase')
    sample = Sample(
        Multilayer(si_layer),
        Multilayer(sio2_layer),
        Multilayer(film_layer),
        Multilayer(superphase),
        name='Film Structure',
    )
    resolution_function = PercentageFwhm(0.02)
    model = Model(sample, 1, 1e-6, resolution_function, 'Film Model')

    # Set some parameters as fittable
    sio2_layer.thickness.fixed = False
    sio2_layer.thickness.bounds = (15, 50)
    film_layer.thickness.fixed = False
    film_layer.thickness.bounds = (200, 300)
    film.sld.fixed = False
    film.sld.bounds = (0.1, 3)

    interface = CalculatorFactory()
    model.interface = interface
    fitter = MultiFitter(model)

    # Capture warnings during fitting
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        analysed = fitter.fit(data)

        # Check that warnings were issued about zero variance points
        fitting_warnings = [str(warning.message) for warning in w
                          if "zero variance during fitting" in str(warning.message)]

        # Should have one warning about the 7 zero variance points (5 + 2)
        assert len(fitting_warnings) == 1, f"Expected 1 warning, got {len(fitting_warnings)}: {fitting_warnings}"
        assert "Masked 7 data point(s)" in fitting_warnings[0], f"Unexpected warning content: {fitting_warnings[0]}"
    # Basic checks that fitting completed despite zero variance points
    assert 'R_0_model' in analysed.keys()
    assert 'SLD_0' in analysed.keys()
    assert 'success' in analysed.keys()
