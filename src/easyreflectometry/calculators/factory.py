"""
Factory for creating reflectometry calculators.

This module provides both the new stateless factory pattern (CalculatorFactory)
and maintains backwards compatibility with the old InterfaceFactoryTemplate pattern.
"""

__author__ = 'github.com/wardsimon'

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from easyscience.fitting.calculators import CalculatorFactoryBase
from easyscience.fitting.calculators import InterfaceFactoryTemplate

from .calculator_base import CalculatorBase


class CalculatorFactory(CalculatorFactoryBase):
    """
    Factory for creating reflectometry calculators.

    This factory follows the new corelib CalculatorFactoryBase pattern, which is
    stateless - it only creates calculators without maintaining state about which
    calculator is "current".

    However, for backwards compatibility with the existing EasyReflectometry code,
    this factory also maintains a "current" calculator instance similar to the
    old InterfaceFactoryTemplate pattern.

    Example usage (new pattern)::

        factory = CalculatorFactory()
        calculator = factory.create('refnx', model=my_model)
        reflectivity = calculator.calculate(q_values)

    Example usage (backwards compatible)::

        factory = CalculatorFactory()
        factory()  # Returns the current calculator instance
        factory.switch('refl1d')
    """

    def __init__(self, calculator_name: Optional[str] = None):
        """Initialize the factory.

        :param calculator_name: Optional name of the default calculator to use.
                                If not provided, uses the first available calculator.
        """
        self._calculator_registry: Dict[str, Type[CalculatorBase]] = {}
        self._current_calculator: Optional[CalculatorBase] = None
        self._current_calculator_name: Optional[str] = None

        # Build registry from CalculatorBase._calculators
        for calc_class in CalculatorBase._calculators:
            name = getattr(calc_class, 'name', calc_class.__name__)
            self._calculator_registry[name] = calc_class

        # Initialize the default calculator
        if len(self._calculator_registry) > 0:
            if calculator_name is not None and calculator_name in self._calculator_registry:
                default_name = calculator_name
            else:
                default_name = list(self._calculator_registry.keys())[0]
            self._current_calculator_name = default_name
            self._current_calculator = self._calculator_registry[default_name]()

    @property
    def available_calculators(self) -> List[str]:
        """Return list of available calculator names.

        :return: List of calculator names.
        """
        return list(self._calculator_registry.keys())

    # Alias for backwards compatibility
    @property
    def available_interfaces(self) -> List[str]:
        """Return list of available calculator names (alias for available_calculators).

        :return: List of calculator names.
        """
        return self.available_calculators

    @property
    def current_interface_name(self) -> str:
        """Return the name of the current calculator.

        :return: Name of the current calculator.
        """
        return self._current_calculator_name

    @property
    def current_interface(self) -> Type[CalculatorBase]:
        """Return the class of the current calculator.

        :return: The calculator class.
        """
        return self._calculator_registry[self._current_calculator_name]

    def create(
        self,
        calculator_name: str,
        model=None,
        instrumental_parameters=None,
        **kwargs,
    ) -> CalculatorBase:
        """Create a new calculator instance.

        This follows the new corelib CalculatorFactoryBase pattern.

        :param calculator_name: Name of the calculator to create.
        :param model: Optional model to associate with the calculator.
        :param instrumental_parameters: Optional instrumental parameters (not used currently).
        :param kwargs: Additional arguments for the calculator.
        :return: A new calculator instance.
        :raises ValueError: If the calculator name is not recognized.
        """
        if calculator_name not in self._calculator_registry:
            available = ', '.join(self.available_calculators)
            raise ValueError(f"Unknown calculator '{calculator_name}'. Available: {available}")

        calculator_class = self._calculator_registry[calculator_name]
        return calculator_class(model=model)

    def switch(self, new_calculator: str, fitter=None) -> None:
        """Switch to a different calculator.

        This is for backwards compatibility with the old InterfaceFactoryTemplate.

        :param new_calculator: Name of the calculator to switch to.
        :param fitter: Optional fitter to update bindings for.
        :raises AttributeError: If the calculator name is not valid.
        """
        if new_calculator not in self._calculator_registry:
            raise AttributeError('The user supplied interface is not valid.')

        self._current_calculator_name = new_calculator
        self._current_calculator = self._calculator_registry[new_calculator]()

        # Update fitter bindings if provided
        if fitter is not None:
            if hasattr(fitter, '_fit_object'):
                obj = getattr(fitter, '_fit_object')
                try:
                    if hasattr(obj, 'update_bindings'):
                        obj.update_bindings()
                except Exception as e:
                    print(f'Unable to auto generate bindings.\n{e}')
            elif hasattr(fitter, 'generate_bindings'):
                try:
                    fitter.generate_bindings()
                except Exception as e:
                    print(f'Unable to auto generate bindings.\n{e}')

    def reset_storage(self) -> None:
        """Reset the storage of the current calculator."""
        if self._current_calculator is not None:
            return self._current_calculator.reset_storage()

    def sld_profile(self, model_id: str) -> tuple:
        """Get the SLD profile from the current calculator.

        :param model_id: The model identifier.
        :return: Tuple of (z, sld) arrays.
        """
        if self._current_calculator is not None:
            return self._current_calculator.sld_profile(model_id)
        return ([], [])

    def generate_bindings(self, model, *args, ifun=None, **kwargs):
        """Generate bindings for a model using the current calculator.

        :param model: The model to generate bindings for.
        """
        if self._current_calculator is None:
            return

        class_links = self._current_calculator.create(model)
        props = model._get_linkable_attributes()
        props_names = [prop.name for prop in props]

        for item in class_links:
            for item_key in item.name_conversion.keys():
                if item_key not in props_names:
                    continue
                idx = props_names.index(item_key)
                prop = props[idx]

                # Get value safely
                if hasattr(prop, 'value_no_call_back'):
                    prop_value = prop.value_no_call_back
                else:
                    prop_value = prop.value

                prop._callback = item.make_prop(item_key)
                prop._callback.fset(prop_value)

    @property
    def fit_func(self) -> Callable:
        """Return the fitting function for the current calculator.

        :return: A callable that computes reflectivity.
        """
        def __fit_func(*args, **kwargs):
            if self._current_calculator is not None:
                return self._current_calculator.reflectivity_profile(*args, **kwargs)
            return None

        return __fit_func

    def __call__(self, *args, **kwargs) -> Optional[CalculatorBase]:
        """Return the current calculator instance.

        This is for backwards compatibility with InterfaceFactoryTemplate.

        :return: The current calculator instance.
        """
        return self._current_calculator

    def __reduce__(self):
        """Support pickling of the factory."""
        return (
            self.__state_restore__,
            (
                self.__class__,
                self.current_interface_name,
            ),
        )

    @staticmethod
    def __state_restore__(cls, interface_str):
        """Restore factory state from pickle."""
        obj = cls()
        if interface_str in obj.available_calculators:
            obj.switch(interface_str)
        return obj

    def __repr__(self) -> str:
        """Return string representation of the factory."""
        return f'{self.__class__.__name__}(current={self._current_calculator_name}, available={self.available_calculators})'
