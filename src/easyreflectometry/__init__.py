from importlib import metadata

from .orso_utils import LoadOrso
from .orso_utils import load_data_from_orso_file
from .project import Project

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = '0.0.0'

__all__ = [
    Project,
    __version__,
    LoadOrso,
    load_data_from_orso_file,
]
