# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for fit result display helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from easyreflectometry.display.fit_display import FitDisplay


class FakePosterior:
    def __init__(self) -> None:
        self.draws_flat = np.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
            ],
        )
        self.param_names = ['thickness', 'sld', 'roughness']
        self.n_draws = 4
        self.n_parameters = 3
        self.n_chains = 2

    def gelman_rubin(self):
        return {'thickness': 1.01, 'sld': 1.02}


class BrokenPosterior(FakePosterior):
    def gelman_rubin(self):
        raise RuntimeError('diagnostic failed')


class FakeModel:
    def get_parameters(self):
        return [
            SimpleNamespace(name='scale', value=1.25, fixed=False),
            SimpleNamespace(name='background', value=0.01, fixed=True),
        ]


def test_results_prints_message_when_no_results(capsys):
    display = FitDisplay(SimpleNamespace())

    result = display.results()

    captured = capsys.readouterr()
    assert result is None
    assert 'No fit or posterior results available' in captured.out


def test_results_renders_classical_fit_and_free_parameters(capsys):
    source = SimpleNamespace(
        _fit_results=[SimpleNamespace(chi2=1.5, n_pars=1)],
        _classical_fit_metrics=[
            {
                'objective_chi2': 1.5,
                'objective_reduced_chi': 0.5,
                'classical_chi2': 2.5,
                'classical_reduced_chi': 0.75,
                'n_classical_points': 12,
            },
        ],
        _models=[FakeModel()],
    )
    display = FitDisplay(source)

    result = display.results()

    captured = capsys.readouterr()
    assert result is None
    assert 'Classical Fit Results' in captured.out
    assert 'Total χ²          : 1.500' in captured.out
    assert 'Fitted Parameters' in captured.out
    assert 'scale' in captured.out
    assert 'background' not in captured.out


def test_results_renders_last_fit_results_fallback(capsys):
    source = SimpleNamespace(
        _last_fit_results=[SimpleNamespace(chi2=3.0, n_pars=0)],
        _last_classical_fit_metrics=None,
    )
    display = FitDisplay(source)

    display.results()

    captured = capsys.readouterr()
    assert 'Total χ²          : 3.000' in captured.out
    assert '(empty table)' in captured.out


def test_results_renders_posterior_summary_and_sampler_settings(capsys):
    source = SimpleNamespace(
        _posterior_results=FakePosterior(),
        _last_sampler_settings={
            'samples': 100,
            'burn': 10,
            'thin': 2,
            'chains': 4,
            'population': 8,
            'seed': 123,
        },
    )
    display = FitDisplay(source)

    result = display.results()

    captured = capsys.readouterr()
    assert result is None
    assert 'Bayesian Sampler Settings' in captured.out
    assert 'Posterior Results' in captured.out
    assert 'Convergence (R-hat)' in captured.out
    assert 'Posterior Parameter Summary' in captured.out
    assert 'thickness' in captured.out


def test_results_ignores_failed_rhat_diagnostic(capsys):
    display = FitDisplay(SimpleNamespace(_last_posterior=BrokenPosterior()))

    display.results()

    captured = capsys.readouterr()
    assert 'Posterior Parameter Summary' in captured.out
    assert 'Convergence (R-hat)' not in captured.out


def test_correlations_prints_message_without_sources(capsys):
    display = FitDisplay(SimpleNamespace())

    result = display.correlations()

    captured = capsys.readouterr()
    assert result is None
    assert 'Correlations unavailable' in captured.out


def test_correlations_renders_thresholded_posterior_matrix(capsys):
    display = FitDisplay(SimpleNamespace(_posterior_results=FakePosterior()))

    result = display.correlations(threshold=0.99, precision=3, show_diagonal=False)

    captured = capsys.readouterr()
    assert result is None
    assert 'Parameter Correlations' in captured.out
    assert 'thickness' in captured.out
    assert '1.000' in captured.out


def test_correlations_handles_single_parameter_posterior(capsys):
    posterior = SimpleNamespace(draws_flat=np.array([[1.0], [2.0]]), param_names=['scale'])
    display = FitDisplay(SimpleNamespace(_posterior_results=posterior))

    result = display.correlations()

    captured = capsys.readouterr()
    assert result is None
    assert 'Correlations unavailable' in captured.out
