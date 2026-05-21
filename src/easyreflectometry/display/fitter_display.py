# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Fitter-level display facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .fit_display import FitDisplay
from .posterior_display import PosteriorDisplay

if TYPE_CHECKING:
    from easyreflectometry.fitting import MultiFitter


class FitterDisplay:
    """Display facade attached to ``MultiFitter``.

    Groups result inspection around tasks: fit results, fit correlations,
    posterior pairs, posterior distributions, posterior predictive
    reflectivity, and posterior predictive SLD.

    :param fitter: The owning ``MultiFitter`` instance.
    :type fitter: MultiFitter
    """

    def __init__(self, fitter: MultiFitter) -> None:
        self.fit: FitDisplay = FitDisplay(fitter)
        self.posterior: PosteriorDisplay = PosteriorDisplay(fitter)
