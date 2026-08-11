# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/wardsimon'
from typing import Any
from typing import Callable

from easyscience.fitting.calculators.interface_factory import InterfaceFactoryTemplate

from easyreflectometry.calculators.calculator_base import CalculatorBase


class CalculatorFactory(InterfaceFactoryTemplate):
    """Factory over every registered calculator.

    Calculators register themselves with :class:`CalculatorBase` as they are
    defined, so the factory needs no discovery logic of its own.
    """

    def __init__(self):
        """Init function."""
        super().__init__(interface_list=CalculatorBase._calculators)

    def generate_bindings(self, model: Any, *args: Any, ifun: Any = None, **kwargs: Any) -> None:
        """Attach a model root to the calculator (the root-only rule).

        Overrides the easyscience template, whose implementation drives the
        legacy ``create()`` / ``ItemContainer`` binding machinery that
        stateless calculators no longer provide. A stateless calculator only
        needs to know the model roots: `BaseCore.generate_bindings` funnels
        every sample-tree object (materials, layers, assemblies) through here
        while propagating the interface, and registering those would fill the
        registry with non-roots — so anything that is not an instance of the
        calculator's ``root_type`` is deliberately ignored.
        """
        calculator = self()
        if isinstance(model, calculator.root_type):
            calculator.set_model(model)

    def reset_storage(self) -> None:
        """Reset storage.

        A no-op for stateless calculators, which have no storage.
        """
        return self().reset_storage()

    def sld_profile(self, model_id: str) -> tuple:
        """Sld profile."""
        return self().sld_profile(model_id)

    @property
    def fit_func(self) -> Callable:
        """Fit func.

        Pass through to the calculator's reflectivity method, which is the fit
        entry point for both calculator generations. The easyscience default
        (delegating to an object-level ``fit_func``) is therefore never used
        here.

        :param x_array: points to be calculated at
        :type x_array: np.ndarray
        :param args: positional arguments for the fitting function
        :type args: Any
        :param kwargs: key/value pair arguments for the fitting function.
        :type kwargs: Any
        :return: points calculated at positional values `x`
        :rtype: np.ndarray
        """

        def __fit_func(*args, **kwargs):
            """Fit func."""
            return self().reflectivity_profile(*args, **kwargs)

        return __fit_func
