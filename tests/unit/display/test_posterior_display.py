# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for posterior display helpers."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from easyreflectometry.display.posterior_display import PosteriorDisplay
from easyreflectometry.display.posterior_display import _resolve_param_name
from easyreflectometry.display.posterior_display import _select_top_parameters


class FakePosterior:
    def __init__(self) -> None:
        self.draws_flat = np.array(
            [
                [1.0, 10.0, 0.1, 8.0],
                [2.0, 20.0, 0.2, 7.0],
                [3.0, 30.0, 0.3, 6.0],
                [4.0, 40.0, 0.4, 5.0],
            ],
        )
        self.param_names = ['thickness', 'sld', 'roughness', 'scale']


class FakeModel:
    pass


def test_get_posterior_raises_when_missing():
    display = PosteriorDisplay(SimpleNamespace())

    with pytest.raises(RuntimeError, match='No posterior results available'):
        display._get_posterior()


def test_get_model_returns_first_model():
    model = FakeModel()
    display = PosteriorDisplay(SimpleNamespace(_models=[model]))

    assert display._get_model() is model


def test_get_model_uses_easy_science_fallback():
    model = FakeModel()
    fitter = SimpleNamespace(_fit_objects=[model])
    display = PosteriorDisplay(SimpleNamespace(easy_science_multi_fitter=fitter))

    assert display._get_model() is model


def test_get_model_raises_when_missing():
    display = PosteriorDisplay(SimpleNamespace(_models=[]))

    with pytest.raises(RuntimeError, match='No model available'):
        display._get_model()


def test_pairs_prints_message_when_no_parameters_match(capsys):
    display = PosteriorDisplay(SimpleNamespace(_posterior_results=FakePosterior()))

    display.pairs(parameters=['missing'])

    captured = capsys.readouterr()
    assert 'No parameters to plot' in captured.out


def test_pairs_selects_parameters_and_calls_corner(monkeypatch):
    import easyreflectometry.analysis.bayesian as bayesian

    called = {}

    def fake_plot_corner(draws, names, **kwargs):
        called['shape'] = draws.shape
        called['names'] = names

    monkeypatch.setattr(bayesian, 'plot_corner', fake_plot_corner)
    display = PosteriorDisplay(SimpleNamespace(_posterior_results=FakePosterior()))

    display.pairs(parameters=['sld', 'scale'])

    assert called == {'shape': (4, 2), 'names': ['sld', 'scale']}


def test_distribution_prints_message_for_missing_parameter(capsys):
    display = PosteriorDisplay(SimpleNamespace(_posterior_results=FakePosterior()))

    display.distribution('missing')

    captured = capsys.readouterr()
    assert 'Parameter' in captured.out
    assert 'not found' in captured.out


def test_distribution_plots_single_parameter(monkeypatch):
    import easyreflectometry.analysis.bayesian as bayesian

    called = {}

    def fake_plot_distribution(draws, names, **kwargs):
        called['shape'] = draws.shape
        called['names'] = names

    monkeypatch.setattr(bayesian, 'plot_distribution', fake_plot_distribution)
    display = PosteriorDisplay(SimpleNamespace(_posterior_results=FakePosterior()))

    display.distribution('thickness')

    assert called == {'shape': (4, 1), 'names': ['thickness']}


def test_reflectivity_requires_data_or_q_values():
    source = SimpleNamespace(_posterior_results=FakePosterior(), _models=[FakeModel()])
    display = PosteriorDisplay(source)

    with pytest.raises(ValueError, match='q_values or data'):
        display.reflectivity()


def test_reflectivity_plots_predictive_band(monkeypatch):
    import matplotlib.pyplot as plt

    import easyreflectometry.analysis.bayesian as bayesian

    captured = {}

    def fake_predictive(draws, param_names, model, q_values, n_samples):
        captured['draws_shape'] = draws.shape
        captured['param_names'] = param_names
        captured['model'] = model
        captured['q_values'] = q_values
        captured['n_samples'] = n_samples
        return np.ones(3), np.zeros(3), np.ones(3) * 2

    monkeypatch.setattr(bayesian, 'posterior_predictive_reflectivity', fake_predictive)
    monkeypatch.setattr(plt, 'show', lambda: None)
    model = FakeModel()
    source = SimpleNamespace(_posterior_results=FakePosterior(), _models=[model])
    display = PosteriorDisplay(source)
    q_values = np.array([0.01, 0.02, 0.03])

    display.reflectivity(q_values=q_values, n_samples=5)

    assert captured['draws_shape'] == (4, 4)
    assert captured['param_names'] == ['thickness', 'sld', 'roughness', 'scale']
    assert captured['model'] is model
    np.testing.assert_array_equal(captured['q_values'], q_values)
    assert captured['n_samples'] == 5
    plt.close('all')


def test_reflectivity_extracts_q_values_from_data(monkeypatch):
    import matplotlib.pyplot as plt

    import easyreflectometry.analysis.bayesian as bayesian

    captured = {}

    def fake_predictive(draws, param_names, model, q_values, n_samples):
        captured['q_values'] = q_values
        return np.ones(2), np.zeros(2), np.ones(2) * 2

    monkeypatch.setattr(bayesian, 'posterior_predictive_reflectivity', fake_predictive)
    monkeypatch.setattr(plt, 'show', lambda: None)
    q_values = np.array([0.01, 0.02])
    data = {
        'coords': {'Qz_0': SimpleNamespace(values=q_values)},
        'data': {'R_0': SimpleNamespace(values=np.array([1.0, 0.5]))},
    }
    source = SimpleNamespace(_posterior_results=FakePosterior(), _models=[FakeModel()])
    display = PosteriorDisplay(source)

    display.reflectivity(data=data, n_samples=3)

    np.testing.assert_array_equal(captured['q_values'], q_values)
    plt.close('all')


def test_sld_profile_plots_predictive_band(monkeypatch):
    import matplotlib.pyplot as plt

    import easyreflectometry.analysis.bayesian as bayesian

    captured = {}

    def fake_predictive(draws, param_names, model, n_samples):
        captured['draws_shape'] = draws.shape
        captured['param_names'] = param_names
        captured['model'] = model
        captured['n_samples'] = n_samples
        return np.array([0.0, 1.0]), np.ones(2), np.zeros(2), np.ones(2) * 2

    monkeypatch.setattr(bayesian, 'posterior_predictive_sld_profile', fake_predictive)
    monkeypatch.setattr(plt, 'show', lambda: None)
    model = FakeModel()
    source = SimpleNamespace(_posterior_results=FakePosterior(), _models=[model])
    display = PosteriorDisplay(source)

    display.sld_profile(n_samples=7)

    assert captured['draws_shape'] == (4, 4)
    assert captured['param_names'] == ['thickness', 'sld', 'roughness', 'scale']
    assert captured['model'] is model
    assert captured['n_samples'] == 7
    plt.close('all')


def test_select_top_parameters_returns_all_when_under_limit():
    draws = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    assert _select_top_parameters(draws, ['a', 'b'], max_parameters=3, threshold=None) == ['a', 'b']


def test_select_top_parameters_ranks_by_correlation_threshold():
    draws = np.array(
        [
            [1.0, 1.0, 5.0, 10.0],
            [2.0, 2.0, 1.0, 11.0],
            [3.0, 3.0, 4.0, 12.0],
            [4.0, 4.0, 2.0, 13.0],
        ],
    )

    selected = _select_top_parameters(draws, ['a', 'b', 'c', 'd'], max_parameters=2, threshold=0.9)

    assert selected == ['a', 'b']


def test_resolve_param_name_handles_supported_identifiers():
    names = ['thickness', 'sld']

    assert _resolve_param_name('thickness', names) == 'thickness'
    assert _resolve_param_name(SimpleNamespace(unique_name='sld'), names) == 'sld'
    assert _resolve_param_name(SimpleNamespace(name='thickness'), names) == 'thickness'
    assert _resolve_param_name('missing', names) is None
    assert _resolve_param_name(SimpleNamespace(name='missing'), names) is None
