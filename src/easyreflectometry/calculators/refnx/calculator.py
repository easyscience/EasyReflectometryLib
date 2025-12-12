"""Refnx calculator implementation for EasyReflectometry."""

__author__ = 'github.com/arm61'

from typing import TYPE_CHECKING
from typing import Optional

from ..calculator_base import CalculatorBase
from .wrapper import RefnxWrapper

if TYPE_CHECKING:
    from easyreflectometry.model import Model


class Refnx(CalculatorBase):
    """
    Calculator for refnx.

    This calculator uses the refnx library to perform reflectometry calculations.

    :param model: Optional model to associate with this calculator.
    """

    name = 'refnx'

    _material_link = {
        'sld': 'real',
        'isld': 'imag',
    }

    _layer_link = {
        'thickness': 'thick',
        'roughness': 'rough',
    }

    _item_link = {
        'repetitions': 'repeats',
    }

    _model_link = {
        'scale': 'scale',
        'background': 'bkg',
    }

    def __init__(self, model: Optional['Model'] = None) -> None:
        """Initialize the Refnx calculator.

        :param model: Optional model to associate with this calculator.
        """
        self._wrapper = RefnxWrapper()
        super().__init__(model=model)
