__author__ = 'github.com/wardsimon'
from typing import Callable

from easyscience.fitting.calculators.interface_factory import InterfaceFactoryTemplate

from easyreflectometry.calculators import CalculatorBase


class CalculatorFactory(InterfaceFactoryTemplate):
    def __init__(self):
        super().__init__(interface_list=CalculatorBase._calculators)

    def reset_storage(self) -> None:
        return self().reset_storage()

    def sld_profile(self, model_id: str) -> tuple:
        return self().sld_profile(model_id)

    @property
    def fit_func(self) -> Callable:
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
            return self().reflectivity_profile(*args, **kwargs)

        return __fit_func
