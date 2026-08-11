# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from ..calculator_base import CalculatorBase
from .stateless_wrapper import Refl1dStatelessWrapper


class Refl1d(CalculatorBase):
    """Calculator for refl1d.

    The refl1d objects are built from the easyscience model on every
    evaluation, so there is no mirrored state and no name-link dictionaries.
    """

    name = 'refl1d'

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper = Refl1dStatelessWrapper()
