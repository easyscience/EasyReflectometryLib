# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/wardsimon'
from typing import Callable

from easyscience.fitting.calculators.interface_factory import InterfaceFactoryTemplate

from easyreflectometry.calculators import CalculatorBase


class CalculatorFactory(InterfaceFactoryTemplate):
    def __init__(self):
        """Init function."""
        super().__init__(interface_list=CalculatorBase._calculators)

    def reset_storage(self) -> None:
        """Reset storage."""
        return self().reset_storage()

    def sld_profile(self, model_id: str) -> tuple:
        """Sld profile."""
        return self().sld_profile(model_id)

    def polarized_reflectivity_profiles(self, x_array, model_id: str) -> dict:
        """Reflectivity profiles of all four spin channels ('pp', 'pm', 'mp', 'mm')."""
        return self().polarized_reflectivity_profiles(x_array, model_id)

    def magnetic_sld_profile(self, model_id: str) -> tuple:
        """Nuclear and magnetic sld profiles: z, sld(z), rhoM(z) and thetaM(z)."""
        return self().magnetic_sld_profile(model_id)

    @property
    def fit_func(self) -> Callable:
        """Fit func."""
        """
        Pass through to the underlying interfaces fitting function.

        :param x_array: points to be calculated at
        :type x_array: np.ndarray
        :param args: positional arguments for the fitting function
        :type args: Any
        :param kwargs: key/value pair arguments for the fitting function.
        :type kwargs: Any
        :return: points calculated at positional values `x`
        :rtype: np.ndarray
        #"""

        def __fit_func(*args, **kwargs):
            """Fit func."""
            return self().reflectity_profile(*args, **kwargs)

        return __fit_func
