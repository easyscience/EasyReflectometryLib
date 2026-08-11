# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import ABCMeta
from typing import Callable

import numpy as np
from easyscience.io import SerializerComponent

from easyreflectometry.model import Model

from .stateless_wrapper_base import StatelessWrapperBase


class CalculatorBase(SerializerComponent, metaclass=ABCMeta):
    """Template for a calculator, and the registry the factory discovers.

    A calculator keeps **no copy** of any parameter. It is handed the model
    roots through :meth:`set_model` and walks them whenever it is asked to
    calculate, so the easyscience ``Parameter`` is the only persistent store of
    a value.

    ``root_type`` is the contract with
    ``easyscience.fitting.calculators.interface_factory.InterfaceFactoryTemplate``:
    it names the class whose instances :meth:`set_model` accepts. The interface
    is propagated to every child of a model, so without it the registry would
    fill up with materials and layers. It lives here, on the object the factory
    holds, not on the wrapper the calculator delegates to.
    """

    _calculators: list[CalculatorBase] = []  # class variable to store all calculators

    root_type = Model

    def __init_subclass__(cls, is_abstract: bool = False, **kwargs) -> None:
        r"""Initialise all subclasses so that they can be created in the factory.

        Parameters
        ----------
        cls :
        is_abstract : bool, optional
            Is this a subclass which shouldn't be added. By default, False.
        **kwargs :
            Key word arguments.
        """
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._calculators.append(cls)

    def __init__(self):
        """Init function."""
        self._namespace = {}
        self._wrapper: StatelessWrapperBase

    # ----- model registry -----

    def set_model(self, model: Model) -> None:
        """Register a model root with the calculator.

        This is a registry add, not a single slot: one ``CalculatorFactory`` is
        routinely shared by every model of a project, and a last-write-wins slot
        would silently serve whichever model bound last.

        Parameters
        ----------
        model : Model
            The model to register, keyed by its ``unique_name``.
        """
        self._wrapper.set_model(model)

    @property
    def models(self) -> dict[str, Model]:
        """The registered models, keyed by ``unique_name``."""
        return self._wrapper.models

    # ----- evaluation -----

    def reflectivity_profile(self, x_array: np.ndarray, model_id: str | None = None) -> np.ndarray:
        """Determine the reflectivity profile for the given range and model.

        This is the fit entry point:
        :attr:`~easyreflectometry.calculators.factory.CalculatorFactory.fit_func`
        routes here, and it evaluates from the *current* state of the registered
        model, which is what makes per-iteration parameter plumbing
        unnecessary.

        Parameters
        ----------
        x_array : np.ndarray
            Points to be calculated at.
        model_id : str | None, optional
            The ``unique_name`` of the model to evaluate. May be omitted when
            exactly one model is registered. By default, None.
        """
        return self._wrapper.calculate(x_array, model_id)

    def reflectity_profile(self, x_array: np.ndarray, model_id: str | None = None) -> np.ndarray:
        """Deprecated alias of :meth:`reflectivity_profile`.

        .. deprecated::
            The method was misspelled; use ``reflectivity_profile``. The alias
            is kept for one release so that external callers keep working.

        Parameters
        ----------
        x_array : np.ndarray
            Points to be calculated at.
        model_id : str | None, optional
            The model id. By default, None.
        """
        warnings.warn(
            '`reflectity_profile` is deprecated (it was a typo), use `reflectivity_profile` instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.reflectivity_profile(x_array, model_id)

    def sld_profile(self, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_id : str | None, optional
            The ``unique_name`` of the model to evaluate. May be omitted when
            exactly one model is registered. By default, None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            z and sld(z).
        """
        return self._wrapper.sld_profile(model_id)

    # ----- configuration -----

    def set_resolution_function(self, resolution_function: Callable[[np.array], np.array]) -> None:
        """Set resolution function.

        A calculator still carries configuration such as the resolution
        function and the magnetism flag; what it does not carry is mirrored
        parameter or structure state.
        """
        return self._wrapper.set_resolution_function(resolution_function)

    @property
    def include_magnetism(self) -> bool:
        """Include magnetism."""
        return self._wrapper.magnetism

    @include_magnetism.setter
    def include_magnetism(self, magnetism: bool) -> None:
        """Set the magnetism flag for the calculator.

        Parameters
        ----------
        magnetism : bool
            True if the calculator should include magnetism.
        """
        self._wrapper.magnetism = magnetism

    # ----- backend to core pull contract -----

    def reconcile(self) -> dict[str, float]:
        """Return the final backend values after a run.

        Write-through backends push every backend-side assignment into the core
        ``Parameter`` as it happens (see
        :class:`~easyreflectometry.calculators.refnx.stateless_wrapper.WriteBackParameter`),
        so nothing can have drifted and there is nothing to reconcile.

        Returns
        -------
        dict[str, float]
            Mapping of ``Parameter.unique_name`` to its final backend value.
            Empty for write-through backends.
        """
        return {}

    # ----- structure mirroring API, no-ops -----
    #
    # The model and sample code calls these unconditionally on every structural
    # edit. A calculator has nothing to mirror: the next evaluation walks the
    # current object graph and sees the change. Implementing them as no-ops
    # keeps the calculator generation out of the model code; the calls
    # themselves are removed at the next major version.

    def reset_storage(self) -> None:
        """No-op, a calculator has no storage to reset."""

    def assign_material_to_layer(self, material_id: str, layer_id: str) -> None:
        """No-op, see the class docstring."""

    def add_layer_to_item(self, layer_id: str, item_id: str) -> None:
        """No-op, see the class docstring."""

    def remove_layer_from_item(self, layer_id: str, item_id: str) -> None:
        """No-op, see the class docstring."""

    def add_item_to_model(self, item_id: str, model_id: str) -> None:
        """No-op, see the class docstring."""

    def remove_item_from_model(self, item_id: str, model_id: str) -> None:
        """No-op, see the class docstring."""
