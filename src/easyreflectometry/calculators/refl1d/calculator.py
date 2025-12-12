"""Refl1d calculator implementation for EasyReflectometry."""

__author__ = 'github.com/arm61'

from typing import TYPE_CHECKING
from typing import Optional

from ..calculator_base import CalculatorBase
from .wrapper import Refl1dWrapper

if TYPE_CHECKING:
    from easyreflectometry.model import Model


class Refl1d(CalculatorBase):
    """
    Calculator for refl1d.

    This calculator uses the refl1d library to perform reflectometry calculations.

    :param model: Optional model to associate with this calculator.
    """

    name = 'refl1d'

    _material_link = {
        'sld': 'rho',
        'isld': 'irho',
    }

    _layer_link = {
        'thickness': 'thickness',
        'roughness': 'interface',
    }

    _item_link = {
        'repetitions': 'repeat',
    }

    _model_link = {
        'scale': 'scale',
        'background': 'bkg',
    }

    def __init__(self, model: Optional['Model'] = None) -> None:
        """Initialize the Refl1d calculator.

        :param model: Optional model to associate with this calculator.
        """
        self._wrapper = Refl1dWrapper()
        super().__init__(model=model)
