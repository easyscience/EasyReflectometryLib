# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Bayesian posterior analysis for reflectometry fitting results."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    import corner as _corner

    _HAS_CORNER = True
except ImportError:
    _HAS_CORNER = False

try:
    import arviz as _arviz

    _HAS_ARVIZ = True
except ImportError:
    _HAS_ARVIZ = False


def _require_corner():
    if not _HAS_CORNER:
        raise ImportError(
            'The ``corner`` library is required for corner plots. '
            'Install it with ``pip install corner`` or '
            '``pip install easyreflectometry[bayesian]``.'
        )


def _require_arviz():
    if not _HAS_ARVIZ:
        raise ImportError(
            'The ``arviz`` library is required for trace plots and R-hat. '
            'Install it with ``pip install arviz`` or '
            '``pip install easyreflectometry[bayesian]``.'
        )


def _to_arviz_data(draws: np.ndarray, param_names: list[str]):
    """Convert posterior draws to an arviz InferenceData object.

    :param draws: Posterior samples, shape ``(n_samples, n_params)`` or
        ``(n_chains, n_draws, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :return: arviz InferenceData object.
    """
    draws = np.asarray(draws)
    if draws.ndim == 2:
        draws = draws[np.newaxis, ...]  # (1, n_samples, n_params)

    # Build a dict of {param_name: (chain, draw) array}
    posterior_dict = {}
    for i, name in enumerate(param_names):
        posterior_dict[name] = draws[:, :, i]

    return _arviz.from_dict({'posterior': posterior_dict})


class PosteriorResults:
    """Container for Bayesian posterior samples with analysis methods.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column of ``draws``).
    :type param_names: list[str]
    :param logp: Log-posterior values, shape ``(n_samples,)``, or ``None``.
    :type logp: np.ndarray | None
    :param sampler_state: Raw sampler state object (e.g. BUMPS ``DreamState``), or ``None``.
    :type sampler_state: Any | None
    """

    def __init__(
        self,
        draws: np.ndarray,
        param_names: list[str],
        logp: np.ndarray | None = None,
        sampler_state: Any | None = None,
    ):
        self.draws = np.asarray(draws)
        self.param_names = list(param_names)
        self.logp = np.asarray(logp) if logp is not None else None
        self.sampler_state = sampler_state

    def __repr__(self) -> str:
        n_samples, n_params = self.draws.shape
        return (
            f'PosteriorResults(n_samples={n_samples}, n_params={n_params}, '
            f'param_names={self.param_names})'
        )

    def summary(self) -> str:
        """Return a formatted summary table with mean, sd, and HDI for each parameter.

        :return: Formatted summary table as a string.
        :rtype: str
        """
        return posterior_summary(self.draws, self.param_names)

    def corner(self, **kwargs) -> None:
        """Plot parameter correlation corner plot.

        Requires the ``corner`` library.

        :param kwargs: Additional keyword arguments passed to ``corner.corner``.
        """
        plot_corner(self.draws, self.param_names, **kwargs)

    def trace(self, **kwargs) -> None:
        """Plot MCMC trace plot.

        Requires the ``arviz`` library.

        :param kwargs: Additional keyword arguments passed to ``arviz.plot_trace``.
        """
        plot_trace(self.draws, self.param_names, **kwargs)

    def credible_interval(self, alpha: float = 0.95) -> dict:
        """Compute equal-tailed credible intervals for each parameter.

        :param alpha: Credible interval width (e.g. 0.95 for 95%).
        :type alpha: float
        :return: Dictionary mapping parameter name to ``(lower, upper)``.
        :rtype: dict
        """
        return credible_intervals(self.draws, self.param_names, alpha=alpha)

    def gelman_rubin(self) -> dict | None:
        """Compute the Gelman-Rubin R-hat convergence diagnostic.

        Requires the ``arviz`` library. Returns ``None`` if ``arviz`` is not
        available.

        :return: Dictionary mapping parameter name to R-hat value, or ``None``.
        :rtype: dict | None
        """
        if not _HAS_ARVIZ:
            warnings.warn(
                'The ``arviz`` library is required for Gelman-Rubin R-hat. '
                'Install it with ``pip install arviz``.',
                UserWarning,
            )
            return None
        # arviz requires at least 2 chains; treat the posterior as one chain
        data = _to_arviz_data(self.draws, self.param_names)
        rhat = _arviz.rhat(data)
        return {name: float(rhat[name].values) for name in self.param_names}


def posterior_summary(draws: np.ndarray, param_names: list[str]) -> str:
    """Return a formatted summary table with mean, sd, and HDI for each parameter.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :return: Formatted summary table as a string.
    :rtype: str
    """
    draws = np.asarray(draws)
    lines = [
        f'{"parameter":<30s} {"mean":>10s} {"sd":>10s} {"hdi_2.5%":>10s} {"hdi_97.5%":>10s}'
    ]
    for i, name in enumerate(param_names):
        col = draws[:, i]
        lo, hi = np.percentile(col, [2.5, 97.5])
        lines.append(f'{name:<30s} {col.mean():>10.4f} {col.std():>10.4f} {lo:>10.4f} {hi:>10.4f}')
    return '\n'.join(lines)


def plot_corner(draws: np.ndarray, param_names: list[str], **kwargs) -> None:
    """Plot a parameter correlation corner plot.

    Requires the ``corner`` library.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param kwargs: Additional keyword arguments passed to ``corner.corner``.
    """
    _require_corner()
    draws = np.asarray(draws)
    defaults = {
        'labels': param_names,
        'quantiles': [0.16, 0.5, 0.84],
        'show_titles': True,
        'title_fmt': '.3f',
        'title_kwargs': {'fontsize': 12},
    }
    defaults.update(kwargs)
    _corner.corner(draws, **defaults)


def plot_trace(draws: np.ndarray, param_names: list[str], **kwargs) -> None:
    """Plot MCMC trace plot.

    Requires the ``arviz`` library.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param kwargs: Additional keyword arguments passed to ``arviz.plot_trace``.
    """
    _require_arviz()
    idata = _to_arviz_data(draws, param_names)
    _arviz.plot_trace(idata, var_names=param_names, **kwargs)


def credible_intervals(
    draws: np.ndarray,
    param_names: list[str],
    alpha: float = 0.95,
) -> dict:
    """Compute equal-tailed credible intervals for each parameter.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param alpha: Credible interval width (e.g. 0.95 for 95%).
    :type alpha: float
    :return: Dictionary mapping parameter name to ``(lower, upper)``.
    :rtype: dict
    """
    draws = np.asarray(draws)
    tail = (1.0 - alpha) / 2.0
    lo_pct = tail * 100
    hi_pct = (1.0 - tail) * 100
    result = {}
    for i, name in enumerate(param_names):
        col = draws[:, i]
        lo, hi = np.percentile(col, [lo_pct, hi_pct])
        result[name] = (float(lo), float(hi))
    return result


def _save_parameter_state(model) -> dict:
    """Save the current values and errors of all free parameters in a model.

    :param model: A reflectometry model with ``get_parameters()``.
    :return: Dictionary mapping ``unique_name`` to ``(value, error)``.
    :rtype: dict
    """
    state = {}
    for param in model.get_parameters():
        state[param.unique_name] = (param.value, param.error)
    return state


def _restore_parameter_state(model, state: dict) -> None:
    """Restore parameter values and errors from a saved state.

    :param model: A reflectometry model with ``get_parameters()``.
    :param state: Dictionary mapping ``unique_name`` to ``(value, error)``.
    """
    for param in model.get_parameters():
        if param.unique_name in state:
            param.value = state[param.unique_name][0]
            param.error = state[param.unique_name][1]


def _apply_draw(model, draws: np.ndarray, param_names: list[str], row: int) -> None:
    """Apply a single posterior draw to the model parameters.

    Parameter lookup uses ``unique_name``, matching the BUMPS names after
    removing the minimizer prefix, which avoids collisions when repeated models
    or multi-contrast fits contain similarly named parameters.

    :param model: A reflectometry model with ``get_parameters()``.
    :param draws: Posterior samples array.
    :param param_names: Parameter names matching the columns of ``draws``.
    :param row: Index of the draw to apply.
    """
    param_lookup = {p.unique_name: p for p in model.get_parameters()}
    for j, name in enumerate(param_names):
        if name in param_lookup:
            param_lookup[name].value = float(draws[row, j])


def posterior_predictive_reflectivity(
    draws: np.ndarray,
    param_names: list[str],
    model,
    q_values: np.ndarray,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the posterior predictive reflectivity with credible intervals.

    Parameter values and errors are saved before applying any posterior draw
    and restored in a ``finally`` block, so the model is not left mutated.

    :param draws: Posterior samples, shape ``(n_samples_posterior, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names matching the columns of ``draws``.
    :type param_names: list[str]
    :param model: A reflectometry model with ``interface.fit_func``.
    :param q_values: Q values at which to evaluate reflectivity.
    :type q_values: np.ndarray
    :param n_samples: Number of posterior draws to use (last ``n_samples``).
    :type n_samples: int
    :return: Tuple of ``(median, lower_95, upper_95)`` reflectivity arrays.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    draws = np.asarray(draws)
    q_values = np.asarray(q_values)

    n_total = draws.shape[0]
    n_use = min(n_samples, n_total)
    sample_indices = range(n_total - n_use, n_total)

    saved_state = _save_parameter_state(model)
    try:
        reflectivity_samples = []
        for i in sample_indices:
            _apply_draw(model, draws, param_names, i)
            r_calc = model.interface.fit_func(q_values, model.unique_name)
            reflectivity_samples.append(np.asarray(r_calc))
    finally:
        _restore_parameter_state(model, saved_state)

    reflectivity_samples = np.array(reflectivity_samples)
    median = np.median(reflectivity_samples, axis=0)
    lower = np.percentile(reflectivity_samples, 2.5, axis=0)
    upper = np.percentile(reflectivity_samples, 97.5, axis=0)
    return median, lower, upper


def posterior_predictive_sld_profile(
    draws: np.ndarray,
    param_names: list[str],
    model,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the posterior predictive SLD profile with credible intervals.

    Parameter values and errors are saved before applying any posterior draw
    and restored in a ``finally`` block, so the model is not left mutated.

    :param draws: Posterior samples, shape ``(n_samples_posterior, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names matching the columns of ``draws``.
    :type param_names: list[str]
    :param model: A reflectometry model with ``interface.sld_profile``.
    :param n_samples: Number of posterior draws to use (last ``n_samples``).
    :type n_samples: int
    :return: Tuple of ``(z, median, lower_95, upper_95)`` SLD profile arrays.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    draws = np.asarray(draws)

    n_total = draws.shape[0]
    n_use = min(n_samples, n_total)
    sample_indices = range(n_total - n_use, n_total)

    saved_state = _save_parameter_state(model)
    try:
        sld_samples = []
        z_shared = None
        for i in sample_indices:
            _apply_draw(model, draws, param_names, i)
            z, sld = model.interface.sld_profile(model.unique_name)
            if z_shared is None:
                z_shared = np.asarray(z)
            sld_samples.append(np.asarray(sld))
    finally:
        _restore_parameter_state(model, saved_state)

    sld_samples = np.array(sld_samples)
    median = np.median(sld_samples, axis=0)
    lower = np.percentile(sld_samples, 2.5, axis=0)
    upper = np.percentile(sld_samples, 97.5, axis=0)
    return z_shared, median, lower, upper
