from importlib import metadata

from .analysis.bayesian import PosteriorResults
from .project import Project

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    Project,
    PosteriorResults,
    __version__,
]
