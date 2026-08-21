# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""EasyReflectometry library."""

from importlib import metadata

from .analysis.bayesian import PosteriorResults
from .constraints import constrain
from .constraints import constrain_equal
from .constraints import unconstrain
from .project import Project

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    'Project',
    'PosteriorResults',
    '__version__',
    'constrain',
    'constrain_equal',
    'unconstrain',
]
