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
    :param chains: Optional number of chains when draws are 3D, or ``None``.
    :type chains: int | None
    """

    def __init__(
        self,
        draws: np.ndarray,
        param_names: list[str],
        logp: np.ndarray | None = None,
        sampler_state: Any | None = None,
        chains: int | None = None,
    ):
        self._draws = np.asarray(draws)
        self.param_names = list(param_names)
        self.logp = np.asarray(logp) if logp is not None else None
        self.sampler_state = sampler_state
        self._chains = chains

    # -- backward-compatible raw-access property ---------------------------------

    @property
    def draws(self) -> np.ndarray:
        """Return the raw draws in the stored shape (2D or 3D).

        Prefer ``draws_flat`` for statistical summaries and plotting.
        """
        return self._draws

    # -- chain-aware accessors --------------------------------------------------

    @property
    def draws_flat(self) -> np.ndarray:
        """Posterior samples flattened to ``(n_samples, n_params)``.

        If the stored draws are 3D ``(n_chains, n_draws_per_chain, n_params)``
        they are flattened to two dimensions.
        """
        if self._draws.ndim == 3:
            return self._draws.reshape(-1, self._draws.shape[2])
        return self._draws

    @property
    def draws_by_chain(self) -> np.ndarray | None:
        """Posterior samples as ``(n_chains, n_draws_per_chain, n_params)``
        when chains are available, otherwise ``None``.
        """
        if self._draws.ndim == 3:
            return self._draws
        return None

    @property
    def n_chains(self) -> int | None:
        """Number of chains when available, otherwise ``None``."""
        if self._chains is not None:
            return self._chains
        if self._draws.ndim == 3:
            return self._draws.shape[0]
        return None

    @property
    def n_draws(self) -> int:
        """Total number of posterior samples (flattened across chains)."""
        return self.draws_flat.shape[0]

    @property
    def n_parameters(self) -> int:
        """Number of parameters in the posterior."""
        return self._draws.shape[-1]

    # -- serialization ----------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> PosteriorResults:
        """Construct from the dictionary returned by ``MultiFitter.sample()``.

        :param d: Dictionary with keys ``'draws'``, ``'param_names'`` and
            optional ``'logp'``, ``'state'``, ``'chains'``.
        :type d: dict
        :return: A new ``PosteriorResults`` instance.
        :rtype: PosteriorResults
        """
        return cls(
            draws=d['draws'],
            param_names=d['param_names'],
            logp=d.get('logp'),
            sampler_state=d.get('state'),
            chains=d.get('chains'),
        )

    def to_dict(self) -> dict:
        """Export to a dictionary compatible with the legacy sampler output.

        :return: Dictionary with keys ``'draws'``, ``'param_names'`` and
            optional ``'logp'``, ``'state'``, ``'chains'``.
        :rtype: dict
        """
        d: dict = {'draws': self._draws, 'param_names': self.param_names}
        if self.logp is not None:
            d['logp'] = self.logp
        if self.sampler_state is not None:
            d['state'] = self.sampler_state
        if self._chains is not None:
            d['chains'] = self._chains
        return d

    # -- repr -------------------------------------------------------------------

    def __repr__(self) -> str:
        draws = self.draws_flat
        n_samples = draws.shape[0]
        n_params = draws.shape[1]
        extras = ''
        if self.n_chains is not None and self.n_chains > 1:
            extras = f', n_chains={self.n_chains}'
        return f'PosteriorResults(n_samples={n_samples}, n_params={n_params}{extras}, param_names={self.param_names})'

    # -- analysis methods (use draws_flat) --------------------------------------

    def summary(self) -> str:
        """Return a formatted summary table with mean, sd, and HDI for each parameter.

        :return: Formatted summary table as a string.
        :rtype: str
        """
        return posterior_summary(self.draws_flat, self.param_names)

    def corner(self, **kwargs) -> None:
        """Plot parameter correlation corner plot.

        Requires the ``corner`` library.

        :param kwargs: Additional keyword arguments passed to ``corner.corner``.
        """
        plot_corner(self.draws_flat, self.param_names, **kwargs)

    def trace(self, **kwargs) -> None:
        """Plot MCMC trace plot.

        Requires the ``arviz`` library.

        :param kwargs: Additional keyword arguments passed to ``arviz.plot_trace``.
        """
        plot_trace(self.draws_flat, self.param_names, **kwargs)

    def credible_interval(self, alpha: float = 0.95) -> dict:
        """Compute equal-tailed credible intervals for each parameter.

        :param alpha: Credible interval width (e.g. 0.95 for 95%).
        :type alpha: float
        :return: Dictionary mapping parameter name to ``(lower, upper)``.
        :rtype: dict
        """
        return credible_intervals(self.draws_flat, self.param_names, alpha=alpha)

    def gelman_rubin(self) -> dict | None:
        """Compute the Gelman-Rubin R-hat convergence diagnostic.

        Requires at least two chains and the ``arviz`` library.
        Returns ``None`` when fewer than two chains are available.

        :return: Dictionary mapping parameter name to R-hat value, or ``None``.
        :rtype: dict | None
        """
        if not _HAS_ARVIZ:
            warnings.warn(
                'The ``arviz`` library is required for Gelman-Rubin R-hat. Install it with ``pip install arviz``.',
                UserWarning,
            )
            return None
        if self.n_chains is None or self.n_chains < 2:
            return None
        draws_for_rhat = self.draws_by_chain if self.draws_by_chain is not None else self._draws
        data = _to_arviz_data(draws_for_rhat, self.param_names)
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


def plot_corner(
    draws: np.ndarray,
    param_names: list[str],
    return_figure: bool = False,
    **kwargs,
):
    """Plot an interactive parameter correlation corner (pairs) plot.

    Uses Plotly when available for interactive rendering; falls back to
    the ``corner`` library for static plots.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param return_figure: When ``True``, return the Plotly ``Figure`` object
        instead of displaying it.  Returns ``None`` when Plotly is unavailable.
    :type return_figure: bool
    :param kwargs: Additional keyword arguments (forwarded to the
        underlying plotting backend). Use ``show=False`` to build the plot
        without displaying it.
    :return: Plotly Figure when ``return_figure=True`` and Plotly is available,
        otherwise ``None``.
    """
    draws = np.asarray(draws)
    if draws.ndim == 3:
        draws = draws.reshape(-1, draws.shape[-1])
    n_params = draws.shape[1]
    show = kwargs.pop('show', not return_figure)

    if n_params < 2:
        _plot_corner_fallback(draws, param_names, show=show, **kwargs)
        return None

    # Prefer Plotly for interactive display
    result = _plot_corner_plotly(draws, param_names, show=show, return_figure=return_figure, **kwargs)
    if result is not False:
        return result if return_figure else None

    # Fall back to matplotlib corner library
    _plot_corner_fallback(draws, param_names, show=show, **kwargs)
    return None


def _plot_corner_plotly(
    draws: np.ndarray,
    param_names: list[str],
    show: bool = True,
    return_figure: bool = False,
    **kwargs,
):
    """Build an interactive Plotly corner/pairs plot.

    Returns the ``Figure`` when ``return_figure=True``, ``True`` on success
    with display, or ``False`` when Plotly is unavailable.
    """
    try:
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    n_params = draws.shape[1]
    labels = [str(n) for n in param_names]

    # Build subplot grid
    fig = make_subplots(
        rows=n_params,
        cols=n_params,
        shared_xaxes='columns',
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    # Compute global ranges per parameter
    ranges = []
    for i in range(n_params):
        col = draws[:, i]
        pad = (col.max() - col.min()) * 0.05
        ranges.append((col.min() - pad, col.max() + pad))

    # Compute correlation matrix for upper triangle annotations
    corr = np.corrcoef(draws, rowvar=False)

    for row in range(n_params):
        for col_idx in range(n_params):
            subplot_ref = (row + 1, col_idx + 1)

            if row == col_idx:
                # Diagonal: histogram
                _add_plotly_diagonal_histogram(
                    fig,
                    draws[:, row],
                    labels[row],
                    subplot_ref,
                    ranges[row],
                )
            elif row > col_idx:
                # Lower triangle: scatter + KDE contours
                _add_plotly_scatter_panel(
                    fig,
                    draws[:, col_idx],
                    draws[:, row],
                    subplot_ref,
                    ranges[col_idx],
                    ranges[row],
                )
            else:
                # Upper triangle: correlation annotation
                _add_plotly_correlation_annotation(
                    fig,
                    corr[row, col_idx],
                    subplot_ref,
                    labels[col_idx],
                    labels[row],
                )

    # Axis labels on the outer edges only
    for i in range(n_params):
        fig.update_xaxes(
            title_text=labels[i],
            row=n_params,
            col=i + 1,
            title_font={'size': 11},
        )
        fig.update_yaxes(
            title_text=labels[i],
            row=i + 1,
            col=1,
            title_font={'size': 11},
        )

    # Hide tick labels on interior axes
    for row in range(1, n_params + 1):
        for col_idx in range(1, n_params + 1):
            if row < n_params:
                fig.update_xaxes(showticklabels=False, row=row, col=col_idx)
            if col_idx > 1:
                fig.update_yaxes(showticklabels=False, row=row, col=col_idx)

    title = kwargs.get('title', 'Posterior Pair Plot')
    fig.update_layout(
        title=title,
        height=250 * n_params,
        width=280 * n_params,
        showlegend=False,
        margin={'l': 60, 'r': 20, 't': 50, 'b': 60},
        template='plotly_white',
    )

    if return_figure:
        return fig
    if show:
        fig.show()
    return True


def plot_distribution(
    draws: np.ndarray,
    param_names: list[str],
    return_figure: bool = False,
    **kwargs,
):
    """Plot one-dimensional marginal posterior distributions.

    Uses Plotly when available for interactive rendering; falls back to
    matplotlib + KDE.

    :param draws: Posterior samples, shape ``(n_samples, n_params)``.
    :type draws: np.ndarray
    :param param_names: Parameter names (one per column).
    :type param_names: list[str]
    :param return_figure: When ``True``, return the Plotly ``Figure`` object
        instead of displaying it.  Returns ``None`` when Plotly is unavailable.
    :type return_figure: bool
    :param kwargs: Forwarded to the underlying backend.
    :return: Plotly Figure when ``return_figure=True`` and Plotly is available,
        otherwise ``None``.
    """
    draws = np.asarray(draws)
    if draws.ndim == 3:
        draws = draws.reshape(-1, draws.shape[-1])
    show = kwargs.pop('show', not return_figure)

    result = _plot_distribution_plotly(draws, param_names, show=show, return_figure=return_figure, **kwargs)
    if result is not False:
        return result if return_figure else None

    _plot_distribution_fallback(draws, param_names, show=show, **kwargs)
    return None


def _plot_distribution_plotly(
    draws: np.ndarray,
    param_names: list[str],
    show: bool = True,
    return_figure: bool = False,
    **kwargs,
):
    """Build a Plotly figure of 1-D marginal histograms + KDE.

    Returns the ``Figure`` when ``return_figure=True``, ``True`` on success, or
    ``False`` when Plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return False

    n = draws.shape[1]
    cols = min(3, n)
    rows = max(1, (n + cols - 1) // cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(param_names[:n]),
        horizontal_spacing=0.10,
        vertical_spacing=0.20,
    )

    # Shrink the auto-generated subplot title annotations so they fit in each cell
    for ann in fig.layout.annotations:
        ann.font = {'size': 11}

    for i, name in enumerate(param_names[:n]):
        row = i // cols + 1
        col_idx = i % cols + 1
        data = draws[:, i]

        mean_val = np.mean(data)
        median_val = np.median(data)
        ci_lo, ci_hi = np.percentile(data, [2.5, 97.5])

        counts, edges = np.histogram(data, bins=30, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bar_width = float(edges[1] - edges[0]) * 0.9

        hover = (
            f'{name}<br>'
            f'value=%{{x:.4f}}<br>'
            f'density=%{{y:.4f}}<br>'
            f'mean={mean_val:.4f}<br>'
            f'median={median_val:.4f}<br>'
            f'95% CI [{ci_lo:.4f}, {ci_hi:.4f}]'
            f'<extra></extra>'
        )

        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                width=bar_width,
                marker_color='rgb(100,149,237)',
                opacity=0.7,
                showlegend=False,
                hovertemplate=hover,
            ),
            row=row,
            col=col_idx,
        )

        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            x_kde = np.linspace(data.min(), data.max(), 200)
            fig.add_trace(
                go.Scatter(
                    x=x_kde,
                    y=kde(x_kde),
                    mode='lines',
                    line={'color': 'rgb(220,80,40)', 'width': 2},
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=row,
                col=col_idx,
            )
        except ImportError:
            pass

        fig.add_shape(
            type='line',
            x0=mean_val,
            x1=mean_val,
            y0=0,
            y1=1,
            yref='y domain',
            line={'dash': 'dash', 'color': 'red', 'width': 1.5},
            opacity=0.7,
            row=row,
            col=col_idx,
        )
        fig.add_shape(
            type='line',
            x0=median_val,
            x1=median_val,
            y0=0,
            y1=1,
            yref='y domain',
            line={'dash': 'dot', 'color': 'green', 'width': 1.5},
            opacity=0.7,
            row=row,
            col=col_idx,
        )

    # Hide unused subplot panels
    for i in range(n, rows * cols):
        row = i // cols + 1
        col_idx = i % cols + 1
        fig.update_xaxes(visible=False, row=row, col=col_idx)
        fig.update_yaxes(visible=False, row=row, col=col_idx)

    title = kwargs.get('title', 'Marginal Posterior Distributions')
    fig.update_layout(
        title=title,
        height=max(320, 300 * rows),
        showlegend=False,
        margin={'l': 60, 'r': 20, 't': 80, 'b': 60},
        template='plotly_white',
    )

    if return_figure:
        return fig
    if show:
        fig.show()
    return True


def _plot_distribution_fallback(
    draws: np.ndarray,
    param_names: list[str],
    show: bool = True,
    **kwargs,
) -> None:
    """Fall back to matplotlib for 1-D marginal distributions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn('Neither Plotly nor matplotlib is available for distribution plots.', UserWarning)
        return

    n = draws.shape[1]
    cols = min(3, n)
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()

    for i, name in enumerate(param_names[:n]):
        ax = axes[i]
        data = draws[:, i]

        ax.hist(data, bins='auto', density=True, alpha=0.6, color='tab:blue', edgecolor='white')
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), color='tab:orange', linewidth=2)
        except ImportError:
            pass

        mean_v = data.mean()
        median_v = float(np.median(data))
        ci_lo, ci_hi = np.percentile(data, [2.5, 97.5])
        ax.axvline(mean_v, color='red', linestyle='--', alpha=0.7, label=f'mean={mean_v:.3f}')
        ax.axvline(median_v, color='green', linestyle=':', alpha=0.7, label=f'median={median_v:.3f}')
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color='gray')
        ax.set_title(name)
        ax.legend(fontsize='small')

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()


def _add_plotly_diagonal_histogram(fig, data, label, subplot_ref, x_range):
    """Add a histogram trace to a diagonal panel."""
    import plotly.graph_objects as go

    mean_val = np.mean(data)
    median_val = np.median(data)
    ci_lo, ci_hi = np.percentile(data, [2.5, 97.5])

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=30,
            marker_color='rgb(100,149,237)',
            opacity=0.7,
            name=label,
            hovertemplate=f'{label}<br>value=%{{x:.4f}}<br>count=%{{y}}<extra></extra>',
        ),
        row=subplot_ref[0],
        col=subplot_ref[1],
    )

    # Mean line — add_vline does not support row/col in subplot figures;
    # use add_shape with yref='y domain' instead.
    fig.add_shape(
        type='line',
        x0=mean_val,
        x1=mean_val,
        y0=0,
        y1=1,
        yref='y domain',
        line={'dash': 'dash', 'color': 'red', 'width': 1},
        opacity=0.5,
        row=subplot_ref[0],
        col=subplot_ref[1],
    )
    # Median line
    fig.add_shape(
        type='line',
        x0=median_val,
        x1=median_val,
        y0=0,
        y1=1,
        yref='y domain',
        line={'dash': 'dot', 'color': 'green', 'width': 1},
        opacity=0.5,
        row=subplot_ref[0],
        col=subplot_ref[1],
    )

    # Annotation with stats
    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref=f'x{subplot_ref[1]}',
        yref=f'y{subplot_ref[1]}',
        text=(f'mean={mean_val:.3f}<br>median={median_val:.3f}<br>95% CI [{ci_lo:.3f}, {ci_hi:.3f}]'),
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font={'size': 9, 'color': '#555555'},
        align='right',
    )

    fig.update_xaxes(range=x_range, row=subplot_ref[0], col=subplot_ref[1])


def _add_plotly_scatter_panel(fig, x_data, y_data, subplot_ref, x_range, y_range):
    """Add scatter + KDE contour traces to an off-diagonal panel."""
    import plotly.graph_objects as go

    # Thin to avoid overcrowding
    n = min(len(x_data), 2000)
    idx = np.random.choice(len(x_data), size=n, replace=False) if len(x_data) > n else slice(None)
    xs, ys = x_data[idx], y_data[idx]

    # Scatter
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker={
                'size': 3,
                'color': 'rgb(100,149,237)',
                'opacity': 0.4,
            },
            hovertemplate='x=%{{x:.4f}}<br>y=%{{y:.4f}}<extra></extra>',
        ),
        row=subplot_ref[0],
        col=subplot_ref[1],
    )

    # KDE contour overlay when scipy is available
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(np.vstack([x_data, y_data]))
        grid_x = np.linspace(x_range[0], x_range[1], 40)
        grid_y = np.linspace(y_range[0], y_range[1], 40)
        xx, yy = np.meshgrid(grid_x, grid_y)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)

        fig.add_trace(
            go.Contour(
                x=grid_x,
                y=grid_y,
                z=zz,
                contours_coloring='lines',
                line_width=1,
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(50,50,50,0.5)']],
                showscale=False,
                hoverinfo='skip',
            ),
            row=subplot_ref[0],
            col=subplot_ref[1],
        )
    except ImportError:
        pass

    fig.update_xaxes(range=x_range, row=subplot_ref[0], col=subplot_ref[1])
    fig.update_yaxes(range=y_range, row=subplot_ref[0], col=subplot_ref[1])


def _add_plotly_correlation_annotation(fig, corr_val, subplot_ref, x_label, y_label):
    """Add a correlation coefficient annotation to an upper-triangle panel."""
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref=f'x{subplot_ref[1]}',
        yref=f'y{subplot_ref[1]}',
        text=f'<b>r = {corr_val:.3f}</b>',
        showarrow=False,
        font={
            'size': 14,
            'color': _correlation_color(corr_val),
        },
        xanchor='center',
        yanchor='middle',
    )


def _correlation_color(val: float) -> str:
    """Return a color representing correlation strength."""
    abs_val = abs(val)
    if abs_val > 0.7:
        return 'rgb(180,0,0)'  # strong red
    if abs_val > 0.4:
        return 'rgb(100,100,100)'  # moderate grey
    return 'rgb(180,180,180)'  # weak light grey


def _plot_corner_fallback(
    draws: np.ndarray,
    param_names: list[str],
    show: bool = True,
    **kwargs,
) -> None:
    """Fall back to the matplotlib ``corner`` library."""
    _require_corner()
    defaults = {
        'labels': param_names,
        'quantiles': [0.16, 0.5, 0.84],
        'show_titles': True,
        'title_fmt': '.3f',
        'title_kwargs': {'fontsize': 12},
    }
    defaults.update(kwargs)
    fig = _corner.corner(draws, **defaults)
    if show:
        fig.show()


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
