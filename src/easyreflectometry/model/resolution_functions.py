# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Resolution functions for the resolution of the experiment.
When a percentage is provided we assume that the resolution is a
Gaussian distribution with a FWHM of the percentage of the q value.
To convert from a sigma value to a FWHM value we use the formula
FWHM = 2.35 * sigma [2 * np.sqrt(2 * np.log(2)) * sigma].
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Union

import numpy as np

DEFAULT_RESOLUTION_FWHM_PERCENTAGE = 5.0


class ResolutionFunction:
    @abstractmethod
    def smearing(self, q: Union[np.array, float]) -> np.array: ...

    @abstractmethod
    def as_dict(self, skip: Optional[List[str]] = None) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> ResolutionFunction:
        """Smearing function."""
        if data['smearing'] == 'PercentageFwhm':
            return PercentageFwhm(data['constant'])
        if data['smearing'] == 'LinearSpline':
            return LinearSpline(data['q_data_points'], data['fwhm_values'])
        if data['smearing'] == 'Pointwise':
            return Pointwise([
                data['q_data_points'],
                data['R_data_points'],
                data['sQz_data_points'],
            ])
        raise ValueError('Unknown resolution function type')


class PercentageFwhm(ResolutionFunction):
    def __init__(self, constant: Union[None, float] = None):
        """Init function."""
        if constant is None:
            constant = DEFAULT_RESOLUTION_FWHM_PERCENTAGE
        self.constant = constant

    def smearing(self, q: Union[np.array, float]) -> np.array:
        """Smearing function."""
        return np.ones(np.array(q).size) * self.constant

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        """As dict."""
        return {'smearing': 'PercentageFwhm', 'constant': self.constant}


class LinearSpline(ResolutionFunction):
    def __init__(self, q_data_points: np.array, fwhm_values: np.array):
        """Init function."""
        self.q_data_points = q_data_points
        self.fwhm_values = fwhm_values

    def smearing(self, q: Union[np.array, float]) -> np.array:
        """Smearing function."""
        return np.interp(q, self.q_data_points, self.fwhm_values)

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        """As dict."""
        return {
            'smearing': 'LinearSpline',
            'q_data_points': list(self.q_data_points),
            'fwhm_values': list(self.fwhm_values),
        }


class Pointwise(ResolutionFunction):
    """Pointwise resolution defined by a per-point resolution provided with the data.

    The resolution is supplied as the variance of the Qz values (``sQz``) at the
    measured Qz data points, which is the form produced by data reduction (e.g.
    ``Qz_0.variances``).  The resolution width at each point is ``sqrt(sQz)``.
    For a requested ``q`` the width is obtained by linearly interpolating onto
    ``q``, exactly as :class:`LinearSpline` does for explicitly provided widths.

    This is a convenience wrapper around :class:`LinearSpline` that derives the
    widths from the ``[Qz, R, sQz]`` triple loaded from a data file; the returned
    widths are consumed by the calculators (refnx ``x_err`` / refl1d ``dq``),
    which perform the actual convolution against the model.
    """

    def __init__(self, q_data_points: List[np.ndarray]):
        """Init function.

        Parameters
        ----------
        q_data_points : List[np.ndarray]
            ``[Qz, R, sQz]`` where ``Qz`` are the measured Qz values, ``R`` the
            measured reflectivity (kept only for serialization round-trips) and
            ``sQz`` the variance of ``Qz`` at each point.
        """
        self.q_data_points = q_data_points

    def smearing(self, q: Optional[Union[np.ndarray, float]] = None) -> np.ndarray:
        """Return the resolution width interpolated onto ``q``.

        The width at each data point is ``sqrt(sQz)``; values are linearly
        interpolated onto the requested ``q``.  When ``q`` is ``None`` the widths
        are returned at the stored data points.
        """
        Qz = np.asarray(self.q_data_points[0], dtype=float)
        sQz = np.asarray(self.q_data_points[2], dtype=float)
        q_eval = Qz if q is None else np.asarray(q, dtype=float)
        widths = np.sqrt(sQz)
        return np.asarray(np.interp(q_eval, Qz, widths))

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        """As dict."""
        return {
            'smearing': 'Pointwise',
            'q_data_points': list(self.q_data_points[0]),
            'R_data_points': list(self.q_data_points[1]),
            'sQz_data_points': list(self.q_data_points[2]),
        }
