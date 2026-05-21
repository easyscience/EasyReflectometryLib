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
        if wrapper is None and self.current_interface_name is not None:
            raise RuntimeError(
                f'Cannot pickle CalculatorFactory: active interface '
                f"{self.current_interface_name!r} exposes no '_wrapper' attribute. "
                'The InterfaceFactoryTemplate API may have changed.'
            )
        return (
            self._state_restore,
            (
                self.__class__,
                self.current_interface_name,
                wrapper.__getstate__() if wrapper is not None else None,
            ),
        )

    @staticmethod
    def _state_restore(cls, interface_str, wrapper_state):
        """Restore a calculator factory with its active wrapper state."""
        obj = cls()
        if interface_str is not None and interface_str in obj.available_interfaces:
            obj.switch(interface_str)
        wrapper = getattr(obj(), '_wrapper', None)
        if wrapper is not None and wrapper_state is not None:
            wrapper.__setstate__(wrapper_state)
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
