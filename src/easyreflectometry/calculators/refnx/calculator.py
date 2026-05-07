# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/arm61'

from ..calculator_base import CalculatorBase
from .wrapper import RefnxWrapper


class Refnx(CalculatorBase):
    """Calculator for refnx."""

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

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper = RefnxWrapper()
