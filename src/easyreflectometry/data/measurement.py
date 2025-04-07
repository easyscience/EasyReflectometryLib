__author__ = 'github.com/arm61'

from typing import TextIO
from typing import Union

import numpy as np
import scipp as sc
from orsopy.fileio import Header
from orsopy.fileio import orso

from easyreflectometry.data import DataSet1D


def load(fname: Union[TextIO, str]) -> sc.DataGroup:
    """Load data from an ORSO .ort file.

    :param fname: The file to be read.
    """
    try:
        return _load_orso(fname)
    except (IndexError, ValueError):
        return _load_txt(fname)


def load_as_dataset(fname: Union[TextIO, str]) -> DataSet1D:
    """Load data from an ORSO .ort file as a DataSet1D."""
    data_group = load(fname)
    return DataSet1D(
        x=data_group['coords']['Qz_0'].values,
        y=data_group['data']['R_0'].values,
        ye=data_group['data']['R_0'].variances,
        xe=data_group['coords']['Qz_0'].variances,
    )


def _load_orso(fname: Union[TextIO, str]) -> sc.DataGroup:
    """Load from an ORSO compatible file.

    :param fname: The path for the file to be read.
    """
    data = {}
    coords = {}
    attrs = {}
    f_data = orso.load_orso(fname)
    for i, o in enumerate(f_data):
        name = i
        if o.info.data_set is not None:
            name = o.info.data_set
        coords[f'Qz_{name}'] = sc.array(
            dims=[f'{o.info.columns[0].name}_{name}'],
            values=o.data[:, 0],
            variances=np.square(o.data[:, 3]),
            unit=sc.Unit(o.info.columns[0].unit),
        )
        try:
            data[f'R_{name}'] = sc.array(
                dims=[f'{o.info.columns[0].name}_{name}'],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
                unit=sc.Unit(o.info.columns[1].unit),
            )
        except TypeError:
            data[f'R_{name}'] = sc.array(
                dims=[f'{o.info.columns[0].name}_{name}'],
                values=o.data[:, 1],
                variances=np.square(o.data[:, 2]),
            )
        attrs[f'R_{name}'] = {'orso_header': sc.scalar(Header.asdict(o.info))}
    return sc.DataGroup(data=data, coords=coords, attrs=attrs)


def _load_txt(fname: Union[TextIO, str]) -> sc.DataGroup:
    """Load data from a simple txt file.

    :param fname: The path for the file to be read.
    """
    f_data = np.loadtxt(fname)
    data = {'R_0': sc.array(dims=['Qz_0'], values=f_data[:, 1], variances=np.square(f_data[:, 2]))}
    coords = {
        data['R_0'].dims[0]: sc.array(
            dims=['Qz_0'],
            values=f_data[:, 0],
            variances=np.square(f_data[:, 3]),
            unit=sc.Unit('1/angstrom'),
        )
    }
    return sc.DataGroup(data=data, coords=coords)
