__author__ = 'github.com/arm61'

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipp as sc
from easyscience.fitting.minimizers.factory import AvailableMinimizers

import easyreflectometry
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.data import DataSet1D
from easyreflectometry.data.measurement import load
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.fitting import _prepare_fit_arrays
from easyreflectometry.fitting import _validate_objective
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
    """Test that zero variance points are handled via Mighell substitution (hybrid default)."""
    import warnings

    import numpy as np

    from easyreflectometry.data.measurement import load

    # Load data that contains zero variance points
    fpath = os.path.join(PATH_STATIC, 'ref_zero_var.txt')

    # First, load the raw data to count zero variance points
    raw_data = np.loadtxt(fpath, delimiter=',', comments='#')
    zero_variance_count = np.sum(raw_data[:, 2] == 0.0)  # Error column
    assert zero_variance_count == 6, f'Expected 6 zero variance points, got {zero_variance_count}'

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

    # Capture warnings during fitting
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        analysed = fitter.fit(data)

        # Under hybrid default, zero variance points trigger Mighell substitution warnings
        mighell_warnings = [str(warning.message) for warning in w if 'Mighell substitution' in str(warning.message)]
        mask_warnings = [str(warning.message) for warning in w if 'Masked' in str(warning.message)]

        # Hybrid mode should NOT produce mask warnings
        assert len(mask_warnings) == 0, f'Unexpected mask warnings under hybrid: {mask_warnings}'

        # If there are zero-variance points in the loaded data, Mighell warnings should appear
        if len(mighell_warnings) > 0:
            for warning_msg in mighell_warnings:
                assert 'zero-variance point(s)' in warning_msg

    # Basic checks that fitting completed
    model_keys = [k for k in analysed.keys() if k.endswith('_model')]
    sld_keys = [k for k in analysed.keys() if k.startswith('SLD_')]
    assert len(model_keys) > 0, f'No model keys found in {list(analysed.keys())}'
    assert len(sld_keys) > 0, f'No SLD keys found in {list(analysed.keys())}'
    assert 'success' in analysed.keys()


def test_fitting_with_manual_zero_variance():
    """Test the fit method with manually created zero variance points using hybrid (default)."""
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
    data = sc.DataGroup(
        {
            'coords': {'Qz_0': sc.array(dims=['Qz_0'], values=qz_values)},
            'data': {'R_0': sc.array(dims=['Qz_0'], values=r_values, variances=variances)},
        }
    )

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
        warnings.simplefilter('always')
        analysed = fitter.fit(data)

        # Under hybrid default, should get Mighell substitution warning, not masking
        mighell_warnings = [str(warning.message) for warning in w if 'Mighell substitution' in str(warning.message)]

        assert len(mighell_warnings) == 1, f'Expected 1 Mighell warning, got {len(mighell_warnings)}: {mighell_warnings}'
        assert '7 zero-variance point(s)' in mighell_warnings[0], f'Unexpected warning content: {mighell_warnings[0]}'

    # Basic checks that fitting completed despite zero variance points
    assert 'R_0_model' in analysed.keys()
    assert 'SLD_0' in analysed.keys()
    assert 'success' in analysed.keys()


def test_fit_single_data_set_1d_masks_zero_variance_points():
    """Legacy mask mode: zero-variance points are dropped."""
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model, objective='legacy_mask')

    captured = {}
    mock_result = MagicMock()
    mock_result.chi2 = 1.0
    mock_result.n_pars = 1

    def _fake_fit(*, x, y, weights):
        captured['x'] = x
        captured['y'] = y
        captured['weights'] = weights
        return [mock_result]

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(side_effect=_fake_fit)

    data = DataSet1D(
        name='single_dataset',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.01, 0.0, 0.04]),
    )

    with pytest.warns(UserWarning, match='Masked 1 data point\\(s\\) in single-dataset fit'):
        result = fitter.fit_single_data_set_1d(data)

    assert result is mock_result
    assert np.allclose(captured['x'][0], np.array([0.01, 0.03]))
    assert np.allclose(captured['y'][0], np.array([1.0, 0.6]))
    assert np.allclose(captured['weights'][0], np.array([10.0, 5.0]))


def test_reduced_chi_uses_global_dof_across_fit_results():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)

    fit_result_1 = MagicMock()
    fit_result_1.chi2 = 10.0
    fit_result_1.x = np.arange(5)
    fit_result_1.n_pars = 3

    fit_result_2 = MagicMock()
    fit_result_2.chi2 = 14.0
    fit_result_2.x = np.arange(7)
    fit_result_2.n_pars = 3

    fitter._fit_results = [fit_result_1, fit_result_2]

    expected = (10.0 + 14.0) / ((5 + 7) - 3)
    assert fitter.reduced_chi == pytest.approx(expected)


def test_fit_single_data_set_1d_all_zero_variance_raises():
    """Legacy mask mode raises when all points have zero variance."""
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model, objective='legacy_mask')

    data = DataSet1D(
        name='all_zero',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.0, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match='all points have zero variance'):
        fitter.fit_single_data_set_1d(data)


def test_chi2_returns_none_before_fit():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)

    assert fitter.chi2 is None


def test_chi2_returns_total_after_fit():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)

    r1 = MagicMock()
    r1.chi2 = 5.0
    r2 = MagicMock()
    r2.chi2 = 3.0

    fitter._fit_results = [r1, r2]
    assert fitter.chi2 == pytest.approx(8.0)


def test_reduced_chi_returns_none_before_fit():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)

    assert fitter.reduced_chi is None


def test_reduced_chi_returns_none_when_dof_zero():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)

    r1 = MagicMock()
    r1.chi2 = 5.0
    r1.x = np.arange(3)
    r1.n_pars = 3  # total_points == n_params => dof == 0

    fitter._fit_results = [r1]
    assert fitter.reduced_chi is None


def test_fit_single_data_set_1d_no_zero_variance():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)

    captured = {}
    mock_result = MagicMock()
    mock_result.chi2 = 2.0
    mock_result.n_pars = 1

    def _fake_fit(*, x, y, weights):
        captured['x'] = x
        captured['y'] = y
        captured['weights'] = weights
        return [mock_result]

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(side_effect=_fake_fit)

    data = DataSet1D(
        name='no_zero',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.01, 0.04, 0.09]),
    )

    result = fitter.fit_single_data_set_1d(data)

    assert result is mock_result
    assert np.allclose(captured['x'][0], np.array([0.01, 0.02, 0.03]))
    assert np.allclose(captured['y'][0], np.array([1.0, 0.8, 0.6]))


# --- New tests for objective-based zero-variance handling ---


def test_objective_validation_rejects_unknown_value():
    with pytest.raises(ValueError, match='Unknown objective'):
        _validate_objective('bad_value')


def test_objective_validation_resolves_auto():
    assert _validate_objective('auto') == 'hybrid'
    assert _validate_objective('hybrid') == 'hybrid'
    assert _validate_objective('legacy_mask') == 'legacy_mask'
    assert _validate_objective('mighell') == 'mighell'


def test_prepare_fit_arrays_weights_always_positive_and_finite():
    """Weights must be strictly positive and finite for all inputs and objectives."""
    test_cases = [
        # (y_vals, variances, description)
        (np.array([0.0]), np.array([0.0]), 'y=0, var=0'),
        (np.array([-0.5]), np.array([0.0]), 'y=-0.5, var=0'),
        (np.array([-1.0]), np.array([0.0]), 'y=-1, var=0'),
        (np.array([1e6]), np.array([0.0]), 'y=1e6, var=0'),
        (np.array([0.5, 0.3, 0.1]), np.array([0.0, 0.0, 0.0]), 'all-zero variances'),
        (np.array([0.5, 0.3, 0.1]), np.array([0.01, 0.0, 0.04]), 'mixed variances'),
        (np.array([0.0, -0.5, -1.0, 1e6]), np.array([0.0, 0.0, 0.0, 0.0]), 'edge y values'),
    ]

    for objective in ('hybrid', 'mighell'):
        for y_vals, variances, desc in test_cases:
            x = np.arange(len(y_vals), dtype=float)
            _, _, weights, _ = _prepare_fit_arrays(x, y_vals, variances, objective)
            assert len(weights) == len(y_vals), f'Wrong length for {desc}, {objective}'
            assert np.all(weights > 0), f'Non-positive weight for {desc}, {objective}: {weights}'
            assert np.all(np.isfinite(weights)), f'Non-finite weight for {desc}, {objective}: {weights}'


def test_prepare_fit_arrays_legacy_mask_drops_zero_variance():
    x = np.array([0.01, 0.02, 0.03])
    y = np.array([1.0, 0.8, 0.6])
    var = np.array([0.01, 0.0, 0.04])

    x_out, y_eff, weights, stats = _prepare_fit_arrays(x, y, var, 'legacy_mask')

    assert np.allclose(x_out, [0.01, 0.03])
    assert np.allclose(y_eff, [1.0, 0.6])
    assert np.allclose(weights, [1.0 / np.sqrt(0.01), 1.0 / np.sqrt(0.04)])
    assert stats == {'valid': 2, 'mighell_substituted': 0, 'masked': 1}


def test_prepare_fit_arrays_hybrid_transforms_zero_variance():
    x = np.array([0.01, 0.02, 0.03])
    y = np.array([1.0, 0.8, 0.6])
    var = np.array([0.01, 0.0, 0.04])

    x_out, y_eff, weights, stats = _prepare_fit_arrays(x, y, var, 'hybrid')

    # x unchanged
    assert np.allclose(x_out, x)
    # Index 0 and 2: standard WLS (unchanged y)
    assert y_eff[0] == pytest.approx(1.0)
    assert y_eff[2] == pytest.approx(0.6)
    assert weights[0] == pytest.approx(1.0 / np.sqrt(0.01))
    assert weights[2] == pytest.approx(1.0 / np.sqrt(0.04))
    # Index 1: Mighell transform — y_eff = y + min(y, 1) = 0.8 + 0.8 = 1.6
    assert y_eff[1] == pytest.approx(0.8 + 0.8)
    # sigma = sqrt(y + 1) = sqrt(1.8)
    assert weights[1] == pytest.approx(1.0 / np.sqrt(1.8))
    assert stats == {'valid': 2, 'mighell_substituted': 1, 'masked': 0}


def test_prepare_fit_arrays_mighell_transforms_all():
    x = np.array([0.01, 0.02])
    y = np.array([0.5, 0.3])
    var = np.array([0.01, 0.04])  # All valid, but mighell transforms everything

    x_out, y_eff, weights, stats = _prepare_fit_arrays(x, y, var, 'mighell')

    assert np.allclose(x_out, x)
    # y_eff = y + min(y, 1) = y + y (since y < 1)
    assert y_eff[0] == pytest.approx(0.5 + 0.5)
    assert y_eff[1] == pytest.approx(0.3 + 0.3)
    # sigma = sqrt(y + 1)
    assert weights[0] == pytest.approx(1.0 / np.sqrt(1.5))
    assert weights[1] == pytest.approx(1.0 / np.sqrt(1.3))
    assert stats == {'valid': 0, 'mighell_substituted': 2, 'masked': 0}


def test_fit_single_data_set_1d_hybrid_keeps_zero_variance_points():
    """Hybrid mode keeps all points (transforms zero-variance ones)."""
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)  # default objective='hybrid'

    captured = {}
    mock_result = MagicMock()
    mock_result.chi2 = 1.0
    mock_result.n_pars = 1

    def _fake_fit(*, x, y, weights):
        captured['x'] = x
        captured['y'] = y
        captured['weights'] = weights
        return [mock_result]

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(side_effect=_fake_fit)

    data = DataSet1D(
        name='hybrid_test',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.01, 0.0, 0.04]),
    )

    with pytest.warns(UserWarning, match='Mighell substitution'):
        result = fitter.fit_single_data_set_1d(data)

    assert result is mock_result
    # All 3 points should be passed through (not masked)
    assert len(captured['x'][0]) == 3
    assert len(captured['y'][0]) == 3
    assert len(captured['weights'][0]) == 3


def test_fit_single_data_set_1d_mighell_warning_mentions_all_points():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model, objective='mighell')

    mock_result = MagicMock()
    mock_result.chi2 = 1.0
    mock_result.reduced_chi = 0.5
    mock_result.n_pars = 1

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(return_value=[mock_result])
    fitter._fit_func = [lambda x: np.zeros_like(x)]

    data = DataSet1D(
        name='mighell_warning',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.01, 0.02, 0.04]),
    )

    with pytest.warns(UserWarning, match=r'Applied Mighell transform to all 3 point\(s\)'):
        fitter.fit_single_data_set_1d(data)


def test_classical_and_objective_chi_are_split_for_fit_results():
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model, objective='mighell')

    fit_result = MagicMock()
    fit_result.chi2 = 0.25
    fit_result.reduced_chi = 0.125
    fit_result.n_pars = 1
    fit_result.x = np.array([0.01, 0.02, 0.03])

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(return_value=[fit_result])
    fitter.easy_science_multi_fitter._fit_objects = [MagicMock(interface=MagicMock())]
    fitter.easy_science_multi_fitter._fit_objects[0].interface.sld_profile.return_value = (np.array([0.0, 1.0]), np.array([1.0, 2.0]))

    fitter._models = [MagicMock(unique_name='model_0', as_dict=MagicMock(return_value={'name': 'model_0'}))]
    fitter._fit_func = [lambda x: np.array([0.8, 0.75, 0.7])]

    data = sc.DataGroup(
        {
            'coords': {'Qz_0': sc.array(dims=['Qz_0'], values=np.array([0.01, 0.02, 0.03]), unit=sc.Unit('1/angstrom'))},
            'data': {'R_0': sc.array(dims=['Qz_0'], values=np.array([1.0, 0.9, 0.7]), variances=np.array([0.01, 0.0, 0.04]))},
            'attrs': {},
        }
    )

    analysed = fitter.fit(data)

    expected_classical_chi2 = ((1.0 - 0.8) / 0.1) ** 2 + ((0.7 - 0.7) / 0.2) ** 2
    expected_classical_reduced = expected_classical_chi2 / (2 - fit_result.n_pars)

    assert analysed['objective_chi2'] == pytest.approx(0.25)
    assert analysed['objective_reduced_chi'] == pytest.approx(0.125)
    assert analysed['classical_chi2'] == pytest.approx(expected_classical_chi2)
    assert analysed['classical_reduced_chi'] == pytest.approx(expected_classical_reduced)
    assert fitter.objective_chi2 == pytest.approx(0.25)
    assert fitter.objective_reduced_chi == pytest.approx(0.125)
    assert fitter.classical_chi2 == pytest.approx(expected_classical_chi2)
    assert fitter.classical_reduced_chi == pytest.approx(expected_classical_reduced)


def test_fit_single_data_set_1d_all_zero_variance_hybrid_does_not_raise():
    """Hybrid mode handles all-zero-variance data without raising."""
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)  # default objective='hybrid'

    captured = {}
    mock_result = MagicMock()
    mock_result.chi2 = 1.0
    mock_result.n_pars = 1

    def _fake_fit(*, x, y, weights):
        captured['x'] = x
        captured['y'] = y
        captured['weights'] = weights
        return [mock_result]

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(side_effect=_fake_fit)

    data = DataSet1D(
        name='all_zero_hybrid',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.0, 0.0, 0.0]),
    )

    with pytest.warns(UserWarning, match='Mighell substitution'):
        result = fitter.fit_single_data_set_1d(data)

    assert result is mock_result
    assert len(captured['x'][0]) == 3


def test_fit_single_data_set_1d_legacy_mask_preserves_old_behavior():
    """Legacy mask mode drops zero-variance points and warns with old message."""
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model, objective='legacy_mask')

    captured = {}
    mock_result = MagicMock()
    mock_result.chi2 = 1.0
    mock_result.n_pars = 1

    def _fake_fit(*, x, y, weights):
        captured['x'] = x
        captured['y'] = y
        captured['weights'] = weights
        return [mock_result]

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(side_effect=_fake_fit)

    data = DataSet1D(
        name='legacy_test',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.01, 0.0, 0.04]),
    )

    with pytest.warns(UserWarning, match='Masked 1 data point'):
        result = fitter.fit_single_data_set_1d(data)

    assert result is mock_result
    assert np.allclose(captured['x'][0], np.array([0.01, 0.03]))
    assert np.allclose(captured['y'][0], np.array([1.0, 0.6]))


def test_fit_multi_dataset_hybrid_uses_transformed_y_and_weights():
    """Multi-dataset fit with hybrid objective transforms zero-variance points."""
    import scipp as sc

    qz_values = np.linspace(0.01, 0.3, 10)
    r_values = np.exp(-qz_values * 50)
    variances = np.ones_like(r_values) * 0.01
    variances[3:5] = 0.0  # 2 zero-variance points

    data = sc.DataGroup(
        {
            'coords': {'Qz_0': sc.array(dims=['Qz_0'], values=qz_values)},
            'data': {'R_0': sc.array(dims=['Qz_0'], values=r_values, variances=variances)},
        }
    )

    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model)

    captured = {}

    def _fake_fit(x, y, weights):
        captured['x'] = x
        captured['y'] = y
        captured['weights'] = weights
        mock_r = MagicMock()
        mock_r.reduced_chi = 1.0
        mock_r.success = True
        mock_r.chi2 = 1.0
        mock_r.n_pars = 1
        mock_r.x = x[0]
        return [mock_r]

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(side_effect=_fake_fit)
    fitter.easy_science_multi_fitter._fit_objects = [MagicMock()]
    fitter.easy_science_multi_fitter._fit_objects[0].interface.sld_profile.return_value = (
        np.linspace(0, 100, 5),
        np.ones(5),
    )

    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fitter.fit(data)

    # All 10 points should be present (not masked)
    assert len(captured['x'][0]) == 10
    assert len(captured['y'][0]) == 10
    assert len(captured['weights'][0]) == 10

    # Zero-variance points should have Mighell-transformed y
    for idx in [3, 4]:
        y_orig = r_values[idx]
        expected_y_eff = y_orig + min(y_orig, 1.0)
        assert captured['y'][0][idx] == pytest.approx(expected_y_eff)

    # Check that Mighell warning was emitted
    mighell_warnings = [str(ww.message) for ww in w if 'Mighell substitution' in str(ww.message)]
    assert len(mighell_warnings) == 1
    assert '2 zero-variance point(s)' in mighell_warnings[0]


def test_fit_warnings_objective_specific():
    """Verify that each objective mode produces the correct warning type."""
    import warnings

    model = Model()
    model.interface = CalculatorFactory()

    mock_result = MagicMock()
    mock_result.chi2 = 1.0
    mock_result.n_pars = 1

    data = DataSet1D(
        name='warn_test',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.01, 0.0, 0.04]),
    )

    for obj, expected_fragment in [
        ('legacy_mask', 'Masked 1 data point(s)'),
        ('hybrid', 'Mighell substitution'),
        ('mighell', 'Mighell substitution'),
    ]:
        fitter = MultiFitter(model, objective=obj)
        fitter.easy_science_multi_fitter = MagicMock()
        fitter.easy_science_multi_fitter.fit = MagicMock(return_value=[mock_result])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fitter.fit_single_data_set_1d(data)

        matching = [str(ww.message) for ww in w if expected_fragment in str(ww.message)]
        assert len(matching) > 0, f'No warning containing {expected_fragment!r} for objective={obj}'


def test_multifitter_constructor_rejects_bad_objective():
    model = Model()
    model.interface = CalculatorFactory()
    with pytest.raises(ValueError, match='Unknown objective'):
        MultiFitter(model, objective='nonsense')


def test_fit_per_call_objective_override():
    """Per-call objective override in fit_single_data_set_1d works."""
    model = Model()
    model.interface = CalculatorFactory()
    fitter = MultiFitter(model, objective='hybrid')  # default

    captured = {}
    mock_result = MagicMock()
    mock_result.chi2 = 1.0
    mock_result.n_pars = 1

    def _fake_fit(*, x, y, weights):
        captured['x'] = x
        captured['y'] = y
        captured['weights'] = weights
        return [mock_result]

    fitter.easy_science_multi_fitter = MagicMock()
    fitter.easy_science_multi_fitter.fit = MagicMock(side_effect=_fake_fit)

    data = DataSet1D(
        name='override_test',
        x=np.array([0.01, 0.02, 0.03]),
        y=np.array([1.0, 0.8, 0.6]),
        ye=np.array([0.01, 0.0, 0.04]),
    )

    # Override to legacy_mask — should drop the zero-variance point
    with pytest.warns(UserWarning, match='Masked 1 data point'):
        fitter.fit_single_data_set_1d(data, objective='legacy_mask')

    assert len(captured['x'][0]) == 2  # one point dropped
