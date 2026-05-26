# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Bayesian posterior analysis for reflectometry fitting results."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    import arviz as _arviz

    _HAS_ARVIZ = True
except ImportError:
    _HAS_ARVIZ = False


def _require_arviz():
    if not _HAS_ARVIZ:
        raise ImportError(
            'The ``arviz`` library is required for trace plots and R-hat. '
            'Install it with ``pip install arviz`` or '
            '``pip install easyreflectometry[bayesian]``.'
        )


def _require_plotly():
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'The ``plotly`` library is required for posterior plots. '
            'Install it with ``pip install plotly`` or '
            '``pip install easyreflectometry[bayesian]``.'
        ) from exc


def _wrap_pair_label(name: str, max_len: int = 16) -> str:
    """Insert ``<br>`` line breaks so long parameter labels don't overlap.

    Dotted names (e.g. ``layer1.thickness``) break on dots; plain names
    word-wrap on spaces at roughly *max_len* characters per line.
    """
    name = name.strip()
    if not name:
        return name
    if '.' in name:
        parts = [p.strip() for p in name.split('.') if p.strip()]
        if parts:
            return '<br>'.join([*(f'{p}.' for p in parts[:-1]), parts[-1]])
    if len(name) <= max_len:
        return name
    words = name.split()
    if len(words) == 1:
        return name
    lines: list[str] = []
    current = ''
    for word in words:
        if current and len(current) + 1 + len(word) > max_len:
            lines.append(current)
            current = word
        else:
            current = f'{current} {word}' if current else word
    if current:
        lines.append(current)
    return '<br>'.join(lines)


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
        return f'PosteriorResults(n_samples={n_samples}, n_params={n_params}, param_names={self.param_names})'

    def summary(self) -> str:
        """Return a formatted summary table with mean, sd, and HDI for each parameter.

        :return: Formatted summary table as a string.
        :rtype: str
        """
        return posterior_summary(self.draws, self.param_names)

    def corner(self) -> Any:
        """Return the parameter-correlation corner plot as a Plotly Figure.

        Requires the ``plotly`` library.

        :return: Plotly Figure.
        """
        return plot_corner(self.draws, self.param_names)

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
                'The ``arviz`` library is required for Gelman-Rubin R-hat. Install it with ``pip install arviz``.',
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
    lines = [f'{"parameter":<30s} {"mean":>10s} {"sd":>10s} {"hdi_2.5%":>10s} {"hdi_97.5%":>10s}']
    for i, name in enumerate(param_names):
        col = draws[:, i]
        lo, hi = np.percentile(col, [2.5, 97.5])
        lines.append(f'{name:<30s} {col.mean():>10.4f} {col.std():>10.4f} {lo:>10.4f} {hi:>10.4f}')
    return '\n'.join(lines)


def plot_corner(draws: np.ndarray, param_names: list[str]) -> Any:
    """Build a parameter-correlation corner plot as a Plotly Figure.

    Marginal densities on the diagonal, posterior scatter and 2-D contour
    overlay on the lower triangle, hidden upper triangle.  Requires
    ``plotly``.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :return: Plotly Figure.
    """
    _require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    draws = np.asarray(draws)
    n_params = len(param_names)

    # Colours mirror easydiffraction's posterior pair plot palette.
    marginal_fill = 'rgba(44, 160, 44, 0.22)'
    marginal_line = 'rgb(44, 160, 44)'
    scatter_color = 'rgba(140, 140, 140, 0.45)'
    contour_colorscale = [
        [0.0, 'rgba(183, 203, 255, 0.94)'],
        [0.35, 'rgba(183, 203, 255, 0.94)'],
        [0.35, 'rgba(138, 169, 252, 0.95)'],
        [0.60, 'rgba(138, 169, 252, 0.95)'],
        [0.60, 'rgba(96, 131, 242, 0.96)'],
        [0.82, 'rgba(96, 131, 242, 0.96)'],
        [0.82, 'rgba(58, 86, 224, 0.98)'],
        [1.0, 'rgba(58, 86, 224, 0.98)'],
    ]

    wrapped_labels = [_wrap_pair_label(name) for name in param_names]

    n_samples = draws.shape[0]
    if n_samples > 1500:
        stride = max(1, n_samples // 1500)
        scatter_draws = draws[::stride]
    else:
        scatter_draws = draws

    fig = make_subplots(
        rows=n_params,
        cols=n_params,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Track which trace types have already been added to the legend.
    legend_shown = {'marginal': False, 'scatter': False, 'contour': False}

    for row in range(n_params):
        for col in range(n_params):
            r, c = row + 1, col + 1
            if col > row:
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)
                continue
            if col == row:
                fig.add_trace(
                    go.Histogram(
                        x=draws[:, row],
                        nbinsx=40,
                        histnorm='probability density',
                        marker=dict(color=marginal_fill, line=dict(color=marginal_line, width=1)),
                        name='Marginal density',
                        legendgroup='marginal',
                        showlegend=not legend_shown['marginal'],
                        hoverinfo='skip',
                    ),
                    row=r,
                    col=c,
                )
                legend_shown['marginal'] = True
            else:
                fig.add_trace(
                    go.Scatter(
                        x=scatter_draws[:, col],
                        y=scatter_draws[:, row],
                        mode='markers',
                        marker=dict(size=3, color=scatter_color),
                        name='Posterior samples',
                        legendgroup='scatter',
                        showlegend=not legend_shown['scatter'],
                        hoverinfo='skip',
                    ),
                    row=r,
                    col=c,
                )
                legend_shown['scatter'] = True
                fig.add_trace(
                    go.Histogram2dContour(
                        x=draws[:, col],
                        y=draws[:, row],
                        ncontours=6,
                        colorscale=contour_colorscale,
                        showscale=False,
                        contours=dict(coloring='lines', showlines=True),
                        line=dict(width=1.2),
                        name='Posterior contours',
                        legendgroup='contour',
                        showlegend=not legend_shown['contour'],
                        hoverinfo='skip',
                    ),
                    row=r,
                    col=c,
                )
                legend_shown['contour'] = True

    # Axis labels: outer edges only (bottom row x-axes, leftmost column y-axes,
    # including the top-left diagonal cell so its parameter is identifiable).
    for i, label in enumerate(wrapped_labels):
        fig.update_xaxes(title_text=label, title_font=dict(size=10), row=n_params, col=i + 1)
        fig.update_yaxes(title_text=label, title_font=dict(size=10), row=i + 1, col=1)
    # Diagonal y-axes are probability density — hide their tick labels (except the
    # top-left, where ticks would be the only cue about the density scale).
    for i in range(1, n_params):
        fig.update_yaxes(showticklabels=False, row=i + 1, col=i + 1)

    fig.update_layout(
        height=max(450, 180 * n_params),
        width=max(550, 180 * n_params + 140),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=1.02,
            font=dict(size=11),
            itemsizing='constant',
        ),
        plot_bgcolor='white',
        margin=dict(l=80, r=160, t=30, b=60),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside')
    fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside')
    return fig


def plot_trace(draws: np.ndarray, param_names: list[str], return_figure: bool = False, **kwargs) -> Any:
    """Plot MCMC trace plot.

    When *return_figure* is ``True`` a Plotly ``Figure`` is returned instead of
    being displayed inline; the caller is responsible for rendering it.  This
    requires the ``plotly`` package.

    :param draws: Posterior samples, shape ``(n_chains, n_draws, n_params)`` or
        ``(n_draws, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param return_figure: Return a Plotly Figure instead of rendering inline.
    :type return_figure: bool
    :param kwargs: Additional keyword arguments passed to ``arviz.plot_trace``
        when *return_figure* is ``False``.
    :return: Plotly Figure when *return_figure* is ``True``, otherwise ``None``.
    """
    draws = np.asarray(draws)
    if draws.ndim == 2:
        draws = draws[np.newaxis, ...]  # (1, n_draws, n_params)
    # draws shape: (n_chains, n_draws, n_params)

    if return_figure:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return None

        n_params = len(param_names)
        n_chains = draws.shape[0]
        fig = make_subplots(
            rows=n_params,
            cols=2,
            column_widths=[0.6, 0.4],
        )
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, name in enumerate(param_names):
            row = i + 1
            show_legend = i == 0
            for c in range(n_chains):
                chain_draws = draws[c, :, i]
                color = colors[c % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        y=chain_draws,
                        mode='lines',
                        line=dict(color=color, width=1),
                        name=f'chain {c}',
                        legendgroup=f'chain {c}',
                        showlegend=show_legend,
                    ),
                    row=row,
                    col=1,
                )
                fig.add_trace(
                    go.Histogram(
                        x=chain_draws,
                        marker_color=color,
                        opacity=0.6,
                        name=f'chain {c}',
                        legendgroup=f'chain {c}',
                        showlegend=show_legend,
                        nbinsx=40,
                    ),
                    row=row,
                    col=2,
                )
            fig.update_yaxes(title_text=name, title_font=dict(size=10), row=row, col=1)
            fig.update_xaxes(title_text=name, title_font=dict(size=10), row=row, col=2)
            fig.update_yaxes(title_text='Count', title_font=dict(size=10), row=row, col=2)
        fig.update_xaxes(title_text='Draw index', title_font=dict(size=10), row=n_params, col=1)
        fig.update_layout(
            height=max(300, 200 * n_params),
            barmode='overlay',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
        )
        return fig

    _require_arviz()
    idata = _to_arviz_data(draws, param_names)
    _arviz.plot_trace(idata, var_names=param_names, **kwargs)
    return None


def plot_distribution(draws: np.ndarray, param_names: list[str], return_figure: bool = False, **kwargs) -> Any:
    """Plot marginal posterior distributions for each parameter.

    When *return_figure* is ``True`` a Plotly ``Figure`` is returned.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param return_figure: Return a Plotly Figure instead of rendering inline.
    :type return_figure: bool
    :param kwargs: Additional keyword arguments (currently unused).
    :return: Plotly Figure when *return_figure* is ``True``, otherwise ``None``.
    """
    draws = np.asarray(draws)
    if draws.ndim == 3:
        draws = draws.reshape(-1, draws.shape[-1])
    # draws shape: (n_samples, n_params)

    if return_figure:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return None

        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols)
        for i, name in enumerate(param_names):
            row = i // n_cols + 1
            col = i % n_cols + 1
            fig.add_trace(
                go.Histogram(
                    x=draws[:, i],
                    name=name,
                    marker_color='#1f77b4',
                    nbinsx=40,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text=name, title_font=dict(size=11), row=row, col=col)
            fig.update_yaxes(title_text='Count', title_font=dict(size=11), row=row, col=col)
        fig.update_layout(
            height=max(300, 250 * n_rows),
            showlegend=False,
        )
        return fig

    return None


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
