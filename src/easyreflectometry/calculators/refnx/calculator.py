# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


from ..calculator_base import CalculatorBase
from .stateless_wrapper import RefnxStatelessWrapper


class Refnx(CalculatorBase):
    """Calculator for refnx.

    The refnx objects are built from the easyscience model on every evaluation,
    and every core ``Parameter`` which reaches refnx is wrapped in a
    write-through proxy, so backend-side assignments come straight back. There
    is no mirrored state and therefore no name-link dictionaries.
    """

    name = 'refnx'

    def __init__(self):
        """Init function."""
        super().__init__()
        self._wrapper = RefnxStatelessWrapper()
