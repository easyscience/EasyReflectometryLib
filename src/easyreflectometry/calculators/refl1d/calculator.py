# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

__author__ = 'github.com/arm61'

from ..calculator_base import CalculatorBase
from .wrapper import Refl1dWrapper


class Refl1d(CalculatorBase):
    """Calculator for refl1."""

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

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper = Refl1dWrapper()
