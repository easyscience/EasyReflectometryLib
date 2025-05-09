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
            return Pointwise(data['q_data_points'])
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
    def __init__(self, q_data_points: np.array):
        self.q_data_points = q_data_points

    def smearing(self, q: Union[np.array, float] = 0.0) -> np.array:

        Qz= self.q_data_points[0]
        R= self.q_data_points[1]
        sR= self.q_data_points[2]
        sQz= self.q_data_points[3]
        smeared = self.apply_smooth_smearing(Qz, R, sR, sQz)
        return smeared

    def as_dict(
        self, skip: Optional[List[str]] = None
    ) -> dict[str, str]:  # skip is kept for consistency of the as_dict signature
        return {'smearing': 'Pointwise', 'q_data_points': list(self.q_data_points)}

    def gaussian_kernel(self, x, sigma):
        """Simple Gaussian kernel function"""
        return np.exp(-x**2/(2*sigma**2))

    def apply_smooth_smearing(self, Qz, R, sR, sQz, n_sigma=3):
        """
        Apply smooth resolution smearing using convolution with Gaussian kernel.
        """
        R_smeared = np.zeros_like(R)
        if not isinstance(Qz, np.ndarray):
            Qz = np.array(Qz)
        if not isinstance(R, np.ndarray):
            R = np.array(R)
        for i, (q, r, sr, sq) in enumerate(zip(Qz, R, sR, sQz)):
            weights = self.gaussian_kernel(Qz - q, sq)
            mask = np.abs(Qz - q) <= n_sigma * sq
            weights[~mask] = 0

            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)

            R_smeared[i] = np.sum(R * weights)
            # Potentially add the pointwise error from sR
            # This can also be used as error bands
            # R_smeared[i] += np.random.normal(0, sr * weights[i])

        return R_smeared