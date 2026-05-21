# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Posterior display — pairs, distributions, predictive checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class PosteriorDisplay:
    """Display posterior distributions and predictive checks.

    :param source: The ``MultiFitter`` or ``Project`` that owns posterior
        state.
    :type source: MultiFitter
    """

    def __init__(self, source) -> None:
        self._source = source

    # -- helpers ----------------------------------------------------------------

    def _get_posterior(self):
        """Return the stored ``PosteriorResults`` or raise."""
        posterior = getattr(self._source, '_posterior_results', None) or getattr(self._source, '_last_posterior', None)
        if posterior is None:
            raise RuntimeError('No posterior results available. Run sample() first.')
        return posterior

    def _get_model(self):
        """Return the first model from the source."""
        # Both MultiFitter and Project have _models
        models = getattr(self._source, '_models', None)
        if models is not None and len(models) > 0:
            return models[0]
        if hasattr(self._source, 'easy_science_multi_fitter'):
            fit_objects = getattr(self._source.easy_science_multi_fitter, '_fit_objects', None)
            if fit_objects:
                return fit_objects[0]
        raise RuntimeError('No model available on fitter/display source.')

    # -- public API -------------------------------------------------------------

    def pairs(
        self,
        parameters: list[str] | None = None,
        threshold: float | None = None,
        max_parameters: int = 6,
        style: str = 'auto',
    ) -> None:
        """Plot posterior pair relationships (corner plot).

        Uses the ``corner`` library when available.  Parameters are
        auto-selected by strongest off-diagonal correlation when not
        explicitly supplied.

        :param parameters: Parameter names to include, or ``None`` for
            auto-selection.
        :param threshold: Minimum absolute correlation for inclusion.
        :param max_parameters: Maximum number of parameters to include.
        :param style: ``'auto'``, ``'fast'``, or ``'full'``.
        """
        posterior = self._get_posterior()
        draws = posterior.draws_flat
        param_names = posterior.param_names

        if parameters is None:
            parameters = _select_top_parameters(draws, param_names, max_parameters, threshold)

        idx = [param_names.index(p) for p in parameters if p in param_names]
        if not idx:
            print('No parameters to plot.')
            return

        from easyreflectometry.analysis.bayesian import plot_corner

        plot_corner(draws[:, idx], [param_names[i] for i in idx])

    def distribution(self, param: object | None = None) -> None:
        """Plot one-dimensional marginal posterior distributions.

        When *param* is ``None``, iterates over all posterior parameters.
        When *param* is supplied, resolves it to a posterior column and
        plots a single distribution.

        :param param: A parameter name (``str``), a ``Parameter`` object,
            or ``None`` to plot all posterior parameters.
        """
        posterior = self._get_posterior()
        draws = posterior.draws_flat
        param_names = posterior.param_names

        # Resolve which parameters to plot
        if param is None:
            to_plot = list(range(len(param_names)))
            labels = param_names
        else:
            name = _resolve_param_name(param, param_names)
            if name is None:
                print(f'Parameter {param!r} not found in posterior parameters: {param_names}')
                return
            idx = param_names.index(name)
            to_plot = [idx]
            labels = [name]

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is required for distribution plots.')
            return

        n = len(to_plot)
        cols = min(3, n)
        rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        axes = axes.flatten()

        for ax_idx, col_idx in enumerate(to_plot):
            ax = axes[ax_idx]
            col = draws[:, col_idx]
            label = labels[col_idx]

            # Histogram
            ax.hist(col, bins='auto', density=True, alpha=0.6, color='tab:blue', edgecolor='white')

            # KDE when SciPy is available
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(col)
                x_range = np.linspace(col.min(), col.max(), 200)
                ax.plot(x_range, kde(x_range), color='tab:orange', linewidth=2)
            except ImportError:
                pass

            # Summary markers
            mean = col.mean()
            median = np.median(col)
            ci_lo, ci_hi = np.percentile(col, [2.5, 97.5])

            ax.axvline(mean, color='red', linestyle='--', alpha=0.7, label=f'mean={mean:.3f}')
            ax.axvline(median, color='green', linestyle=':', alpha=0.7, label=f'median={median:.3f}')
            ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='gray', label=f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')

            ax.set_title(label)
            ax.legend(fontsize='small', loc='upper right')

        # Hide unused axes
        for ax_idx in range(n, len(axes)):
            axes[ax_idx].set_visible(False)

        fig.tight_layout()
        plt.show()

    def reflectivity(
        self,
        data: object = None,
        q_values: np.ndarray | None = None,
        n_samples: int = 200,
    ) -> None:
        """Plot posterior-predictive reflectivity with 95 % credible band.

        :param data: ``sc.DataGroup`` with measured data for overlay.
        :param q_values: Q values at which to evaluate reflectivity.
            Required when *data* is not supplied.
        :param n_samples: Number of posterior draws to use.
        """
        posterior = self._get_posterior()
        model = self._get_model()

        from easyreflectometry.analysis.bayesian import posterior_predictive_reflectivity

        if q_values is None and data is not None:
            refl_nums = [k[3:] for k in data['coords'].keys() if 'Qz' == k[:2]]
            q_values = np.asarray(data['coords'][f'Qz_{refl_nums[0]}'].values) if refl_nums else None

        if q_values is None:
            raise ValueError('q_values or data must be supplied.')

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
            if data is not None:
                refl_nums = [k[3:] for k in data['coords'].keys() if 'Qz' == k[:2]]
                for i in refl_nums:
                    r_data_vals = np.asarray(data['data'][f'R_{i}'].values)
                    qz_vals = np.asarray(data['coords'][f'Qz_{i}'].values)
                    plt.semilogy(qz_vals, r_data_vals, 'o', label='Data', alpha=0.6)
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

    def sld_profile(self, n_samples: int = 200) -> None:
        """Plot posterior-predictive SLD profile with 95 % credible band.

        :param n_samples: Number of posterior draws to use.
        """
        posterior = self._get_posterior()
        model = self._get_model()

        from easyreflectometry.analysis.bayesian import posterior_predictive_sld_profile

        z, sld_median, sld_lower, sld_upper = posterior_predictive_sld_profile(
            posterior.draws_flat,
            posterior.param_names,
            model,
            n_samples=n_samples,
        )

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.plot(z, sld_median, label='Posterior median SLD')
            plt.fill_between(z, sld_lower, sld_upper, alpha=0.3, label='95% credible interval')
            plt.xlabel('z / Å')
            plt.ylabel('SLD / 10⁻⁶ Å⁻²')
            plt.title('SLD Profile with Bayesian Uncertainty')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print('matplotlib is required for predictive plots.')


# ==========================================================================
# Internal helpers
# ==========================================================================


def _select_top_parameters(
    draws: np.ndarray,
    param_names: list[str],
    max_parameters: int,
    threshold: float | None,
) -> list[str]:
    """Select parameters with strongest off-diagonal correlations."""
    n = draws.shape[1]
    if n <= max_parameters:
        return param_names

    corr = np.corrcoef(draws, rowvar=False)
    # Score each parameter by maximum absolute off-diagonal correlation
    scores = []
    for i in range(n):
        off_diag = [abs(corr[i, j]) for j in range(n) if i != j]
        scores.append(max(off_diag) if off_diag else 0.0)

    # Sort by descending score
    ranked = sorted(zip(scores, param_names), key=lambda x: x[0], reverse=True)
    if threshold is not None:
        ranked = [(s, name) for s, name in ranked if s >= threshold]

    return [name for _, name in ranked[:max_parameters]]


def _resolve_param_name(param: object, param_names: list[str]) -> str | None:
    """Resolve a parameter identifier to a posterior column name.

    :param param: A ``str`` name, or an object with ``.name`` or
        ``.unique_name`` attributes.
    :param param_names: Available posterior parameter names.
    :return: The matching name, or ``None``.
    """
    if isinstance(param, str):
        if param in param_names:
            return param
        return None

    # Try common attribute names
    for attr in ('unique_name', 'name'):
        val = getattr(param, attr, None)
        if isinstance(val, str) and val in param_names:
            return val

    return None
