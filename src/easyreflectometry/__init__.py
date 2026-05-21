# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""EasyReflectometry library."""

from importlib import metadata

from .analysis.bayesian import PosteriorResults
from .display import FitDisplay
from .display import FitterDisplay
from .display import PosteriorDisplay
from .display import ProjectDisplay
from .project import Project

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    'FitDisplay',
    'FitterDisplay',
    'PosteriorDisplay',
    'PosteriorResults',
    'Project',
    'ProjectDisplay',
    __version__,
]
