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
        if data['smearing'] == 'PercentageFwhm':
            return PercentageFwhm(data['constant'])
        if data['smearing'] == 'LinearSpline':
            return LinearSpline(data['q_data_points'], data['fwhm_values'])
        if data['smearing'] == 'Pointwise':
            return Pointwise([data['q_data_points'], data['R_data_points'], data['sQz_data_points']])
        raise ValueError('Unknown resolution function type')


class PercentageFwhm(ResolutionFunction):
    def __init__(self, constant: Union[None, float] = None):
        if constant is None:
            constant = DEFAULT_RESOLUTION_FWHM_PERCENTAGE
        self.constant = constant

    def smearing(self, q: Union[np.array, float]) -> np.array:
        return np.ones(np.array(q).size) * self.constant

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        return {'smearing': 'PercentageFwhm', 'constant': self.constant}


class LinearSpline(ResolutionFunction):
    def __init__(self, q_data_points: np.array, fwhm_values: np.array):
        self.q_data_points = q_data_points
        self.fwhm_values = fwhm_values

    def smearing(self, q: Union[np.array, float]) -> np.array:
        return np.interp(q, self.q_data_points, self.fwhm_values)

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        return {'smearing': 'LinearSpline', 'q_data_points': list(self.q_data_points), 'fwhm_values': list(self.fwhm_values)}

# add pointwise smearing funtion
class Pointwise(ResolutionFunction):
    def __init__(self, q_data_points: list[np.ndarray]):
        self.q_data_points = q_data_points
        self.q = None

    def smearing(self, q: Union[np.ndarray, float] = None) -> np.ndarray:
        Qz = self.q_data_points[0]
        R = self.q_data_points[1]
        sQz = self.q_data_points[2]
        if q is None:
            q = self.q_data_points[0]
        self.q = q
        sQzs = np.sqrt(sQz)
        if isinstance(Qz, float):
            Qz = np.array(Qz)

        smeared = self.apply_smooth_smearing(Qz, R, sQzs)
        return smeared

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        return {
            'smearing': 'Pointwise',
            'q_data_points': list(self.q_data_points[0]),
            'R_data_points': list(self.q_data_points[1]),
            'sQz_data_points': list(self.q_data_points[2]),
        }

    def gaussian_smearing(self, qt, Qz, R, sQz):
        weights = np.exp(-0.5 * ((qt - Qz) / sQz) ** 2)
        if np.sum(weights) == 0 or not np.isfinite(np.sum(weights)):
            return np.sum(R)
        weights /= sQz * np.sqrt(2 * np.pi)
        return np.sum(R * weights) / np.sum(weights)

    def apply_smooth_smearing(self, Qz, R, sQzs):
        """
        Apply smooth resolution smearing using convolution with Gaussian kernel.
        """
        if self.q is None:
            R_smeared = np.zeros_like(Qz)
        else:
            R_smeared = np.zeros_like(self.q)

        if not isinstance(Qz, np.ndarray):
            Qz = np.array(Qz)
        if not isinstance(R, np.ndarray):
            R = np.array(R)
        R_smeared = np.zeros_like(self.q)

        for i, qt in enumerate(self.q):
            R_smeared[i] = self.gaussian_smearing(qt, Qz, R, sQzs)

        return R_smeared
