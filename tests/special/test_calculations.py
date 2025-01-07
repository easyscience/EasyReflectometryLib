__author__ = 'github.com/arm61'

import unittest

from numpy.testing import assert_almost_equal

from easyreflectometry.special.calculations import area_per_molecule_to_scattering_length_density
from easyreflectometry.special.calculations import molecular_weight
from easyreflectometry.special.calculations import neutron_scattering_length
from easyreflectometry.special.calculations import weighted_average


class TestMaterialMixture(unittest.TestCase):
    def test_weighted_average(self) -> None:
        a = weighted_average(1, 2, 0.5)
        assert_almost_equal(a, 1.5)

    def test_neutron_scattering_length_H2O(self) -> None:
        a = neutron_scattering_length('H2O')
        assert_almost_equal(a.real, -1.6768e-05)
        assert_almost_equal(a.imag, 0.0)

    def test_neutron_scattering_length_B(self) -> None:
        a = neutron_scattering_length('B')
        assert_almost_equal(a.real, 5.3e-05)
        assert_almost_equal(a.imag, -2.1e-06)

    def test_molecular_weight(self) -> None:
        a = molecular_weight('H2O')
        assert_almost_equal(a, 18.015)

    def test_area_per_molecule_to_sld(self) -> None:
        a = area_per_molecule_to_scattering_length_density(2, 1, 0.5)
        assert_almost_equal(a, 4e6)
