# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Fit result display — classical and Bayesian results."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .tables import render_table

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from easyreflectometry.analysis.bayesian import PosteriorResults


class FitDisplay:
    """Display fit results from a ``MultiFitter`` or ``Project``.

    :param source: The ``MultiFitter`` or ``Project`` instance that owns
        fit and posterior state.
    :type source: MultiFitter
    """

    def __init__(self, source) -> None:
        self._source = source

    # -- normalized accessors (work with both MultiFitter and Project) ----------

    def _get_fit_results(self):
        return getattr(self._source, '_fit_results', None) or getattr(self._source, '_last_fit_results', None)

    def _get_posterior(self):
        return getattr(self._source, '_posterior_results', None) or getattr(self._source, '_last_posterior', None)

    def _get_sampler_settings(self):
        return getattr(self._source, '_last_sampler_settings', None)

    def _get_classical_fit_metrics(self):
        return getattr(self._source, '_classical_fit_metrics', None) or getattr(
            self._source, '_last_classical_fit_metrics', None
        )

    def _get_models(self):
        """Return models from source (both MultiFitter and Project have _models)."""
        return getattr(self._source, '_models', None)

    def results(self) -> object | None:
        """Render one or more tables summarising the latest fit or sampler run.

        For a classical fit the table shows minimizer, chi² values, data
        points, free parameters, and fitted parameter values with
        uncertainties.

        For a Bayesian fit the table additionally shows sampler settings,
        posterior shape, convergence diagnostics, and posterior parameter
        summaries.

        :return: A pandas ``DataFrame`` in notebooks, ``None`` in terminals.
        :rtype: object | None
        """
        fit_results = self._get_fit_results()
        posterior = self._get_posterior()
        sampler = self._get_sampler_settings()

        has_fit = fit_results is not None and len(fit_results) > 0
        has_posterior = posterior is not None

        if not has_fit and not has_posterior:
            print('No fit or posterior results available. Run fit() or sample() first.')
            return None

        # -- Fit info table ----------------------------------------------------
        if has_fit:
            self._show_fit_info(fit_results)

        # -- Bayesian table ----------------------------------------------------
        if has_posterior:
            self._show_posterior_info(posterior, sampler)

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _show_fit_info(self, fit_results: list) -> None:
        """Render classical fit metrics and parameter table."""
        metrics = self._get_classical_fit_metrics()
        total_chi2 = sum(float(r.chi2) for r in fit_results)
        n_pars = fit_results[0].n_pars if fit_results else 0

        # Build info rows
        rows = []
        if metrics:
            for i, m in enumerate(metrics):
                rows.append({
                    'dataset': i,
                    'objective_chi2': f'{m["objective_chi2"]:.3f}',
                    'objective_red_chi2': f'{m["objective_reduced_chi"]:.3f}',
                    'classical_chi2': f'{m["classical_chi2"]:.3f}',
                    'classical_red_chi2': f'{m["classical_reduced_chi"]:.3f}',
                    'n_data_points': m.get('n_classical_points', ''),
                })

        cols = ['dataset', 'objective_chi2', 'objective_red_chi2', 'classical_chi2', 'classical_red_chi2', 'n_data_points']
        print('\n' + '=' * 72)
        print('  Classical Fit Results')
        print('=' * 72)
        print(f'  Total χ²          : {total_chi2:.3f}')
        print(f'  Free parameters   : {n_pars}')
        print('-' * 72)
        render_table(rows, cols)

        # Fitted parameter table
        try:
            models = self._get_models()
            if models is not None and len(models) > 0:
                model = models[0]
                param_rows = []
                for p in model.get_parameters():
                    if not p.fixed:
                        param_rows.append({
                            'parameter': p.name,
                            'value': f'{p.value:.6g}',
                            'fixed': str(p.fixed),
                        })
                if param_rows:
                    print('\n  Fitted Parameters')
                    print('-' * 72)
                    render_table(param_rows, ['parameter', 'value', 'fixed'])
        except Exception:
            logger.exception('Failed to render fitted parameter table')

    def _show_posterior_info(
        self,
        posterior: PosteriorResults,
        sampler: dict | None,
    ) -> None:
        """Render Bayesian sampler settings and posterior parameter summary."""
        # -- Sampler settings --------------------------------------------------
        if sampler:
            print('\n' + '=' * 72)
            print('  Bayesian Sampler Settings')
            print('=' * 72)
            print(f'  Samples         : {sampler.get("samples", "")}')
            print(f'  Burn-in         : {sampler.get("burn", "")}')
            print(f'  Thinning        : {sampler.get("thin", "")}')
            print(f'  Chains          : {sampler.get("chains", "")}')
            print(f'  Population      : {sampler.get("population", "")}')
            print(f'  Seed            : {sampler.get("seed", "")}')

        # -- Posterior shape ---------------------------------------------------
        print('\n' + '=' * 72)
        print('  Posterior Results')
        print('=' * 72)
        print(f'  Total samples   : {posterior.n_draws}')
        print(f'  Parameters      : {posterior.n_parameters}')
        if posterior.n_chains is not None:
            print(f'  Chains          : {posterior.n_chains}')

        # -- Convergence diagnostics (R-hat) -----------------------------------
        try:
            rhat = posterior.gelman_rubin()
            if rhat:
                print('-' * 72)
                print('  Convergence (R-hat)')
                print('-' * 72)
                rhat_rows = [{'parameter': name, 'r_hat': f'{value:.4f}'} for name, value in rhat.items()]
                render_table(rhat_rows, ['parameter', 'r_hat'])
        except Exception:
            logger.exception('Failed to compute or render R-hat diagnostics')
        print('-' * 72)
        print('  Posterior Parameter Summary')
        print('-' * 72)
        draws = posterior.draws_flat
        summary_rows = []
        for i, name in enumerate(posterior.param_names):
            col = draws[:, i]
            lo, hi = np.percentile(col, [2.5, 97.5])
            summary_rows.append({
                'parameter': name,
                'mean': f'{col.mean():.6g}',
                'sd': f'{col.std():.6g}',
                'hdi_2.5%': f'{lo:.6g}',
                'hdi_97.5%': f'{hi:.6g}',
            })
        render_table(
            summary_rows,
            ['parameter', 'mean', 'sd', 'hdi_2.5%', 'hdi_97.5%'],
        )

    def correlations(
        self,
        threshold: float | None = None,
        precision: int = 2,
        *,
        max_parameters: int = 6,
        show_diagonal: bool = True,
    ) -> object | None:
        """Render a correlation table (and optionally a heatmap).

        Correlation source priority:

        1. Posterior sample correlation matrix when available.
        2. Classical covariance-derived correlation when surfaced by the
           minimizer backend.
        3. A message that correlations are unavailable.

        :param threshold: Absolute value below which correlations are hidden.
        :type threshold: float | None
        :param precision: Decimal places for correlation values.
        :type precision: int
        :param max_parameters: Cap the number of parameters shown.
        :type max_parameters: int
        :param show_diagonal: Include the diagonal entries (always 1.0).
        :type show_diagonal: bool
        :return: A pandas ``DataFrame`` or ``None``.
        :rtype: object | None
        """
        posterior = self._get_posterior()
        corr = None
        param_names = []

        # Try posterior first
        if posterior is not None:
            draws = posterior.draws_flat
            if draws.shape[1] >= 2:
                corr = np.corrcoef(draws, rowvar=False)
                param_names = posterior.param_names
        else:
            print('Correlations unavailable: no posterior samples and no covariance matrix exposed by the minimizer.')
            return None

        if corr is None:
            print('Correlations unavailable.')
            return None

        rows = []
        for i in range(len(param_names)):
            row = {'parameter': param_names[i]}
            for j in range(len(param_names)):
                if not show_diagonal and i == j:
                    row[param_names[j]] = ''
                else:
                    val = corr[i, j]
                    if threshold is not None and abs(val) < threshold and i != j:
                        row[param_names[j]] = ''
                    else:
                        row[param_names[j]] = f'{val:.{precision}f}'
            rows.append(row)

        cols = ['parameter'] + param_names
        print('\n' + '=' * 72)
        print('  Parameter Correlations')
        print('=' * 72)
        return render_table(rows, cols)
