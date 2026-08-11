# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import ResolutionFunction

if TYPE_CHECKING:
    from easyreflectometry.model import Model


class StatelessWrapperBase:
    """Backend-agnostic part of a stateless wrapper.

    Holds the model registry and the configuration which is genuinely the
    calculator's own (resolution function, magnetism flag). It deliberately
    holds no parameter or structure state: those are built from the model
    inside :meth:`calculate` and thrown away when it returns.
    """

    def __init__(self):
        """Constructor."""
        self._models: dict[str, Model] = {}
        self._magnetism = False
        self._resolution_function = PercentageFwhm()

    # ----- registry -----

    @property
    def models(self) -> dict[str, Model]:
        """The registered models, keyed by ``unique_name``."""
        return self._models

    def set_model(self, model: Model) -> None:
        """Register a model root.

        Parameters
        ----------
        model : Model
            The model to register.
        """
        self._models[model.unique_name] = model

    def _get_model(self, model_id: str | None = None) -> Model:
        """Select a registered model.

        Parameters
        ----------
        model_id : str | None, optional
            The ``unique_name`` of the model. May be omitted when exactly one
            model is registered. By default, None.

        Raises
        ------
        ValueError
            If no model is attached, if several are attached and none was
            selected, or if ``model_id`` is unknown.
        """
        if not self._models:
            raise ValueError(
                'No model is attached to the calculator. Attach one with '
                '`model.interface = calculator_factory` (or `model.generate_bindings()`) '
                'before calculating.'
            )
        if model_id is None:
            if len(self._models) > 1:
                raise ValueError(
                    f'Several models are attached to the calculator ({sorted(self._models)}); '
                    'pass `model_id` to select the one to evaluate.'
                )
            return next(iter(self._models.values()))
        try:
            return self._models[model_id]
        except KeyError:
            raise ValueError(
                f'No model with unique name {model_id!r} is attached to the calculator. '
                f'Attached models: {sorted(self._models)}.'
            ) from None

    # ----- calculation -----

    @abstractmethod
    def calculate(self, q_array: np.ndarray, model_id: str | None = None) -> np.ndarray:
        """For a given q array calculate the corresponding reflectivity.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_id : str | None, optional
            The ``unique_name`` of the model to evaluate. By default, None.

        Returns
        -------
        np.ndarray
            Reflectivity calculated at q.
        """
        ...

    @abstractmethod
    def sld_profile(self, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_id : str | None, optional
            The ``unique_name`` of the model to evaluate. By default, None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            z and sld(z).
        """
        ...

    # ----- configuration -----

    def set_resolution_function(self, resolution_function: ResolutionFunction) -> None:
        """Set the resolution function for the calculator.

        Parameters
        ----------
        resolution_function : ResolutionFunction
            The resolution function.
        """
        self._resolution_function = resolution_function

    @property
    def magnetism(self) -> bool:
        """Magnetism function."""
        return self._magnetism

    @magnetism.setter
    def magnetism(self, magnetism: bool) -> None:
        """Set the magnetism flag.

        Parameters
        ----------
        magnetism : bool
            The magnetism flag.
        """
        self._magnetism = magnetism
