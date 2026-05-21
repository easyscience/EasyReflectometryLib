# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for display facade objects."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use('Agg')

from easyreflectometry.display import FitDisplay
from easyreflectometry.display import FitterDisplay
from easyreflectometry.display import PosteriorDisplay
from easyreflectometry.display import ProjectDisplay
from easyreflectometry.display.project_display import ProjectPosteriorDisplay


class FakePosterior:
    def __init__(self) -> None:
        self.draws_flat = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.param_names = ['thickness', 'sld']


class FakeModel:
    pass


def test_fitter_display_exposes_fit_and_posterior_facades():
    fitter = SimpleNamespace()

    display = FitterDisplay(fitter)

    assert isinstance(display.fit, FitDisplay)
    assert isinstance(display.posterior, PosteriorDisplay)


def test_project_display_exposes_project_facades():
    project = SimpleNamespace()

    display = ProjectDisplay(project)

    assert display._project is project
    assert isinstance(display.fit, FitDisplay)
    assert isinstance(display.posterior, ProjectPosteriorDisplay)


def test_project_posterior_get_model_uses_current_project_model():
    model = FakeModel()
    project = SimpleNamespace(models=[FakeModel(), model], current_model_index=1)
    display = ProjectPosteriorDisplay(project)

    assert display._get_model() is model


def test_project_posterior_reflectivity_resolves_experiment(monkeypatch):
    import matplotlib.pyplot as plt

    import easyreflectometry.analysis.bayesian as bayesian

    model = FakeModel()
    experiment = SimpleNamespace(x=np.array([0.01, 0.02]), y=np.array([1.0, 0.5]))
    captured = {}

    def fake_resolve_experiment(expt_name):
        captured['expt_name'] = expt_name
        return experiment

    def fake_predictive(draws, param_names, model_arg, q_values, n_samples):
        captured['draws_shape'] = draws.shape
        captured['param_names'] = param_names
        captured['model'] = model_arg
        captured['q_values'] = q_values
        captured['n_samples'] = n_samples
        return np.ones(2), np.zeros(2), np.ones(2) * 2

    monkeypatch.setattr(bayesian, 'posterior_predictive_reflectivity', fake_predictive)
    monkeypatch.setattr(plt, 'show', lambda: None)
    project = SimpleNamespace(
        _last_posterior=FakePosterior(),
        models=[model],
        current_model_index=0,
        _resolve_experiment=fake_resolve_experiment,
    )
    display = ProjectPosteriorDisplay(project)

    display.reflectivity(expt_name='experiment-1', n_samples=9)

    assert captured['expt_name'] == 'experiment-1'
    assert captured['draws_shape'] == (2, 2)
    assert captured['param_names'] == ['thickness', 'sld']
    assert captured['model'] is model
    np.testing.assert_array_equal(captured['q_values'], experiment.x)
    assert captured['n_samples'] == 9
    plt.close('all')
