# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum


class PolarizationChannel(str, Enum):
    """Spin cross-section channels for polarized neutron reflectometry.

    The accepted spellings are exactly the enum values ('pp', 'pm', 'mp', 'mm');
    uppercase strings are rejected.
    """

    PP = 'pp'  # non-spin-flip, up-up
    PM = 'pm'  # spin-flip, up-down
    MP = 'mp'  # spin-flip, down-up
    MM = 'mm'  # non-spin-flip, down-down


# Mapping to refl1d cross-section indices: PolarizedNeutronProbe.xs is documented as
# "a sequence pp, pm, mp and mm". The pp/mm assignment is additionally pinned by
# physics tests; pm vs mp rests on the refl1d docstring alone (they are identical by
# symmetry for non-chiral, non-absorptive samples).
POLARIZATION_CHANNEL_TO_INDEX = {
    PolarizationChannel.PP: 0,
    PolarizationChannel.PM: 1,
    PolarizationChannel.MP: 2,
    PolarizationChannel.MM: 3,
}
