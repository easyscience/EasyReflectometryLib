# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Post-hoc analysis utilities for reflectometry fitting results."""

from easyreflectometry.analysis.bayesian import PosteriorResults
from easyreflectometry.analysis.bayesian import credible_intervals
from easyreflectometry.analysis.bayesian import plot_corner
from easyreflectometry.analysis.bayesian import plot_trace
from easyreflectometry.analysis.bayesian import posterior_predictive_reflectivity
from easyreflectometry.analysis.bayesian import posterior_predictive_sld_profile
from easyreflectometry.analysis.bayesian import posterior_summary

__all__ = [
    'PosteriorResults',
    'plot_corner',
    'plot_trace',
    'posterior_summary',
    'credible_intervals',
    'posterior_predictive_reflectivity',
    'posterior_predictive_sld_profile',
]
