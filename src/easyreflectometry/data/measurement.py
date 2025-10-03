__author__ = 'github.com/arm61'

import os
import warnings
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
    basename = os.path.splitext(os.path.basename(fname))[0]
    data_name = 'R_' + basename
    coords_name = 'Qz_' + basename
    coords_name = list(data_group['coords'].keys())[0] if coords_name not in data_group['coords'] else coords_name
    data_name = list(data_group['data'].keys())[0] if data_name not in data_group['data'] else data_name

    # Get the data arrays
    x_vals = data_group['coords'][coords_name].values
    y_vals = data_group['data'][data_name].values
    ye_vals = data_group['data'][data_name].variances
    xe_vals = data_group['coords'][coords_name].variances

    # Check for zero variances and remove those points
    # Only check ye_vals for zero if xe_vals was all zeros (3-column file)
    if np.all(xe_vals == 0.0):
        zero_variance_mask = (ye_vals == 0.0)
    else:
        zero_variance_mask = (ye_vals == 0.0) | (xe_vals == 0.0)
    num_zero_variance = np.sum(zero_variance_mask)

    if num_zero_variance > 0:
        warnings.warn(
            f"Removed {num_zero_variance} data point(s) from '{fname}' due to zero variance.",
            UserWarning
        )
        # Keep only points with non-zero variances
        valid_mask = ~zero_variance_mask
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        ye_vals = ye_vals[valid_mask]
        xe_vals = xe_vals[valid_mask]

    return DataSet1D(
        x=x_vals,
        y=y_vals,
        ye=ye_vals,
        xe=xe_vals,
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

    # Check for zero variances and remove those points
    total_removed = 0
    for key in list(data.keys()):
        data_variances = data[key].variances
        # Find the corresponding coordinate key for this data
        coord_key = None
        for coord_name in coords.keys():
            if data[key].dims == coords[coord_name].dims:
                coord_key = coord_name
                break

        if coord_key is None:
            continue  # Skip if no matching coordinate found

        coord_variances = coords[coord_key].variances

        # Find points with zero variances
        zero_variance_mask = (data_variances == 0.0) | (coord_variances == 0.0)
        num_zero_variance = np.sum(zero_variance_mask)
        total_removed += num_zero_variance

        if num_zero_variance > 0:
            # Keep only points with non-zero variances
            valid_mask = ~zero_variance_mask
            data[key] = sc.array(
                dims=data[key].dims,
                values=data[key].values[valid_mask],
                variances=data[key].variances[valid_mask],
                unit=data[key].unit if hasattr(data[key], 'unit') else None
            )
            coords[coord_key] = sc.array(
                dims=coords[coord_key].dims,
                values=coords[coord_key].values[valid_mask],
                variances=coords[coord_key].variances[valid_mask],
                unit=coords[coord_key].unit
            )

    if total_removed > 0:
        warnings.warn(
            f"Removed {total_removed} data point(s) from '{fname}' due to zero variance.",
            UserWarning
        )

    return sc.DataGroup(data=data, coords=coords, attrs=attrs)


def _load_txt(fname: Union[TextIO, str]) -> sc.DataGroup:
    """Load data from a simple txt file.

    :param fname: The path for the file to be read.
    """
    # fname can have either a space or a comma as delimiter
    # Determine the delimiter used in the file
    delimiter = None
    with open(fname, 'r') as f:
        # find first non-comment and non-empty line
        for line in f:
            if line.strip() and not line.startswith('#'):
                break
        first_line = line
    if ',' in first_line:
        delimiter = ','

    basename = os.path.splitext(os.path.basename(fname))[0]

    try:
        # First load only the data to check column count
        data = np.loadtxt(fname, delimiter=delimiter, comments='#')
        if data.ndim == 1:
            # Handle single row case
            num_columns = len(data)
        else:
            num_columns = data.shape[1]

        # Verify minimum column requirement
        if num_columns < 3:
            raise ValueError(f'File must contain at least 3 columns (found {num_columns})')

        # Now unpack the data based on column count
        if num_columns >= 4:
            x, y, e, xe = np.loadtxt(fname, delimiter=delimiter, comments='#', unpack=True)
        else:  # 3 columns
            x, y, e = np.loadtxt(fname, delimiter=delimiter, comments='#', unpack=True)
            xe = np.zeros_like(x)

    except (ValueError, IOError) as error:
        # Re-raise with more descriptive message
        raise ValueError(f'Failed to load data from {fname}: {str(error)}') from error

    data_name = 'R_' + basename
    coords_name = 'Qz_' + basename

    # Calculate variances
    y_variances = np.square(e)
    x_variances = np.square(xe)

    # Check for zero variances and remove those points
    # Only check y_variances for zero if xe was all zeros (3-column file)
    if np.all(xe == 0.0):
        zero_variance_mask = (y_variances == 0.0)
    else:
        zero_variance_mask = (y_variances == 0.0) | (x_variances == 0.0)
    num_zero_variance = np.sum(zero_variance_mask)

    if num_zero_variance > 0:
        warnings.warn(
            f"Removed {num_zero_variance} data point(s) from '{fname}' due to zero variance.",
            UserWarning
        )
        # Keep only points with non-zero variances
        valid_mask = ~zero_variance_mask
        x = x[valid_mask]
        y = y[valid_mask]
        y_variances = y_variances[valid_mask]
        x_variances = x_variances[valid_mask]

    data = {data_name: sc.array(dims=[coords_name], values=y, variances=y_variances)}
    coords = {
        data[data_name].dims[0]: sc.array(
            dims=[coords_name],
            values=x,
            variances=x_variances,
            unit=sc.Unit('1/angstrom'),
        )
    }
    return sc.DataGroup(data=data, coords=coords)


def merge_datagroups(*data_groups: sc.DataGroup) -> sc.DataGroup:
    """Merge multiple DataGroups into a single DataGroup."""
    merged_data = {}
    merged_coords = {}
    merged_attrs = {}

    for group in data_groups:
        for key, value in group['data'].items():
            if key not in merged_data:
                merged_data[key] = value
            else:
                merged_data[key] = sc.concatenate([merged_data[key], value])

        for key, value in group['coords'].items():
            if key not in merged_coords:
                merged_coords[key] = value
            else:
                merged_coords[key] = sc.concatenate([merged_coords[key], value])

        if 'attrs' not in group:
            continue
        for key, value in group['attrs'].items():
            if key not in merged_attrs:
                merged_attrs[key] = value
            else:
                merged_attrs[key] = {**merged_attrs[key], **value}

    return sc.DataGroup(data=merged_data, coords=merged_coords, attrs=merged_attrs)
