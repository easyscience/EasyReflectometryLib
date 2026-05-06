# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Bayesian analysis module."""

import numpy as np
import pytest


@pytest.fixture
def sample_draws():
    """Generate synthetic posterior draws for testing."""
    rng = np.random.default_rng(42)
    n_samples = 100
    # Two parameters: 'thickness' and 'sld'
    thickness = rng.normal(loc=250, scale=10, size=n_samples)
    sld = rng.normal(loc=2.0, scale=0.2, size=n_samples)
    draws = np.column_stack([thickness, sld])
    param_names = ['Film_thickness', 'Film_sld']
    return draws, param_names


class TestPosteriorSummary:
    def test_returns_string(self, sample_draws):
        from easyreflectometry.analysis.bayesian import posterior_summary

        draws, param_names = sample_draws
        result = posterior_summary(draws, param_names)
        assert isinstance(result, str)
        assert 'parameter' in result
        assert 'mean' in result
        assert 'sd' in result

    def test_contains_param_names(self, sample_draws):
        from easyreflectometry.analysis.bayesian import posterior_summary

        draws, param_names = sample_draws
        result = posterior_summary(draws, param_names)
        for name in param_names:
            assert name in result


class TestCredibleIntervals:
    def test_returns_dict(self, sample_draws):
        from easyreflectometry.analysis.bayesian import credible_intervals

        draws, param_names = sample_draws
        result = credible_intervals(draws, param_names)
        assert isinstance(result, dict)
        for name in param_names:
            assert name in result
            lo, hi = result[name]
            assert lo < hi

    def test_alpha_95_coverage(self, sample_draws):
        from easyreflectometry.analysis.bayesian import credible_intervals

        draws, param_names = sample_draws
        result = credible_intervals(draws, param_names, alpha=0.95)
        for i, name in enumerate(param_names):
            lo, hi = result[name]
            # 95% interval should contain at least 90% of samples
            col = draws[:, i]
            inside = np.sum((col >= lo) & (col <= hi))
            assert inside / len(col) >= 0.90

    def test_alpha_50_narrower(self, sample_draws):
        from easyreflectometry.analysis.bayesian import credible_intervals

        draws, param_names = sample_draws
        ci_95 = credible_intervals(draws, param_names, alpha=0.95)
        ci_50 = credible_intervals(draws, param_names, alpha=0.50)
        for name in param_names:
            assert (ci_95[name][1] - ci_95[name][0]) > (ci_50[name][1] - ci_50[name][0])


class TestPosteriorResults:
    def test_repr(self, sample_draws):
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        rep = repr(pr)
        assert 'PosteriorResults' in rep
        assert str(draws.shape[0]) in rep

    def test_summary_delegates(self, sample_draws):
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        summary_str = pr.summary()
        assert isinstance(summary_str, str)
        assert 'parameter' in summary_str

    def test_credible_interval_delegates(self, sample_draws):
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        ci = pr.credible_interval(alpha=0.95)
        assert isinstance(ci, dict)
        for name in param_names:
            assert name in ci


class TestPosteriorPredictiveReflectivity:
    def test_returns_tuples(self, sample_draws):
        """Test with a mock model that returns a constant array."""
        from unittest.mock import MagicMock

        from easyreflectometry.analysis.bayesian import posterior_predictive_reflectivity

        draws, param_names = sample_draws
        mock_model = MagicMock()
        mock_model.unique_name = 'test_model'
        mock_model.interface = MagicMock()
        mock_model.interface.fit_func = MagicMock(return_value=np.ones(50))
        mock_model.get_parameters = MagicMock(return_value=[])

        q_values = np.linspace(0.01, 0.3, 50)
        median, lower, upper = posterior_predictive_reflectivity(
            draws,
            param_names,
            mock_model,
            q_values,
            n_samples=20,
        )
        assert median.shape == (50,)
        assert lower.shape == (50,)
        assert upper.shape == (50,)


class TestPosteriorPredictiveSLDProfile:
    def test_returns_tuples(self, sample_draws):
        """Test with a mock model that returns constant z and sld."""
        from unittest.mock import MagicMock

        from easyreflectometry.analysis.bayesian import posterior_predictive_sld_profile

        draws, param_names = sample_draws
        mock_model = MagicMock()
        mock_model.unique_name = 'test_model'
        mock_model.interface = MagicMock()
        mock_model.interface.sld_profile = MagicMock(return_value=(np.linspace(0, 500, 100), np.ones(100) * 2.0))
        mock_model.get_parameters = MagicMock(return_value=[])

        z, median, lower, upper = posterior_predictive_sld_profile(
            draws,
            param_names,
            mock_model,
            n_samples=20,
        )
        assert z.shape == (100,)
        assert median.shape == (100,)
        assert lower.shape == (100,)
        assert upper.shape == (100,)


class TestCornerPlot:
    def test_plot_corner_does_not_crash(self, sample_draws):
        """Test that plot_corner does not crash when corner is available."""
        import matplotlib

        matplotlib.use('Agg')  # Non-interactive backend for testing

        try:
            from easyreflectometry.analysis.bayesian import plot_corner

            draws, param_names = sample_draws
            plot_corner(draws, param_names)
        except ImportError:
            pytest.skip('corner library not installed')


class TestSaveRestoreParameterState:
    def test_save_and_restore(self):
        """Test that parameter state save/restore works correctly."""
        from easyreflectometry.analysis.bayesian import _restore_parameter_state
        from easyreflectometry.analysis.bayesian import _save_parameter_state

        # Use simple objects that support attribute assignment
        class MockParam:
            def __init__(self, unique_name, raw_value, error):
                self.unique_name = unique_name
                self.value = raw_value
                self.error = error

        param1 = MockParam('param_a', 1.5, 0.1)
        param2 = MockParam('param_b', 3.0, 0.2)

        class MockModel:
            def get_parameters(self):
                return [param1, param2]

        model = MockModel()

        state = _save_parameter_state(model)
        assert state['param_a'] == (1.5, 0.1)
        assert state['param_b'] == (3.0, 0.2)

        # Modify values
        param1.raw_value = 99.0
        param1.value = 99.0
        param2.raw_value = 99.0
        param2.value = 99.0

        _restore_parameter_state(model, state)
        assert param1.value == 1.5
        assert param1.error == 0.1
        assert param2.value == 3.0
        assert param2.error == 0.2


class TestApplyDraw:
    def test_apply_draw_updates_parameters(self):
        """Test that _apply_draw sets parameter values correctly."""
        from easyreflectometry.analysis.bayesian import _apply_draw

        class MockParam:
            def __init__(self, unique_name):
                self.unique_name = unique_name
                self.value = None

        param_a = MockParam('thickness')
        param_b = MockParam('sld')

        class MockModel:
            def get_parameters(self):
                return [param_a, param_b]

        model = MockModel()
        draws = np.array([[250.0, 2.0], [260.0, 2.1]])
        param_names = ['thickness', 'sld']

        _apply_draw(model, draws, param_names, row=0)
        assert param_a.value == 250.0
        assert param_b.value == 2.0

        _apply_draw(model, draws, param_names, row=1)
        assert param_a.value == 260.0
        assert param_b.value == 2.1
