# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Project-level display facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .fit_display import FitDisplay
from .posterior_display import PosteriorDisplay

if TYPE_CHECKING:
    from easyreflectometry.project import Project


class ProjectDisplay:
    """Display facade attached to ``Project``.

    Groups result inspection around tasks: fit results, fit correlations,
    posterior pairs, posterior distributions, posterior predictive
    reflectivity, and posterior predictive SLD.

    :param project: The owning ``Project`` instance.
    :type project: Project
    """

    def __init__(self, project: Project) -> None:
        self._project = project
        self.fit: FitDisplay = FitDisplay(project)
        self.posterior: ProjectPosteriorDisplay = ProjectPosteriorDisplay(project)


class ProjectPosteriorDisplay(PosteriorDisplay):
    """Project-aware posterior display that resolves experiments by name.

    :param project: The owning ``Project`` instance.
    :type project: Project
    """

    def __init__(self, project: Project) -> None:
        super().__init__(project)
        self._project = project

    def reflectivity(self, expt_name: str | None = None, n_samples: int = 200) -> None:
        """Plot posterior-predictive reflectivity with 95 % credible band.

        :param expt_name: Experiment name, or ``None`` for the current
            experiment.
        :param n_samples: Number of posterior draws to use.
        """
        posterior = self._get_posterior()
        model = self._get_model()
        experiment = self._project._resolve_experiment(expt_name)

        q_values = np.asarray(experiment.x)
        r_data_vals = np.asarray(experiment.y)

        from easyreflectometry.analysis.bayesian import posterior_predictive_reflectivity

        r_median, r_lower, r_upper = posterior_predictive_reflectivity(
            posterior.draws_flat,
            posterior.param_names,
            model,
            q_values,
            n_samples=n_samples,
        )

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(9, 6))
            plt.semilogy(q_values, r_data_vals, 'o', label='Data', alpha=0.6)
            plt.semilogy(q_values, r_median, '-', color='tab:orange', label='Posterior median')
            plt.fill_between(q_values, r_lower, r_upper, color='tab:orange', alpha=0.3, label='95% credible interval')
            plt.xlabel('Q / Å⁻¹')
            plt.ylabel('Reflectivity')
            plt.title('Bayesian Posterior-Predictive Check')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print('matplotlib is required for predictive plots.')

    def _get_model(self):
        """Return the current project model."""
        return self._project.models[self._project.current_model_index]
