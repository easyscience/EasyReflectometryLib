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

    def __reduce__(self):
        """Serialize the active calculator state for worker processes."""
        wrapper = getattr(self(), '_wrapper', None)
        wrapper_state = None
        if wrapper is not None:
            wrapper_state = {
                'storage': wrapper.storage,
                'resolution_function': wrapper._resolution_function,
                'magnetism': wrapper._magnetism,
            }
        return (
            self.__state_restore__,
            (
                self.__class__,
                self.current_interface_name,
                wrapper_state,
            ),
        )

    @staticmethod
    def __state_restore__(cls, interface_str, wrapper_state):
        """Restore a calculator factory with its active wrapper state."""
        obj = cls()
        if interface_str in obj.available_interfaces:
            obj.switch(interface_str)
        wrapper = getattr(obj(), '_wrapper', None)
        if wrapper is not None and wrapper_state is not None:
            wrapper.storage = wrapper_state['storage']
            wrapper._resolution_function = wrapper_state['resolution_function']
            wrapper._magnetism = wrapper_state['magnetism']
        return obj

    def reset_storage(self) -> None:
        """Reset storage."""
        return self().reset_storage()

    def sld_profile(self, model_id: str) -> tuple:
        """Sld profile."""
        return self().sld_profile(model_id)

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
