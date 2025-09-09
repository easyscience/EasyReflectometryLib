import unittest

import numpy as np
from easyscience import global_object
from numpy.testing import assert_almost_equal

from easyreflectometry.sample.elements.materials.material_density import MaterialDensity


class TestMaterialDensity(unittest.TestCase):
    def test_default(self):
        p = MaterialDensity()
        assert p.name == 'EasyMaterialDensity'
        assert p.interface is None
        assert p.density.display_name == 'density'
        assert str(p.density.unit) == 'kg/L'
        assert p.density.value == 2.33
        assert p.density.min == 0
        assert p.density.max == np.inf
        assert p.density.fixed is True

    def test_default_constraint(self):
        p = MaterialDensity()
        assert p.density.value == 2.33
        assert_almost_equal(p.sld.value, 2.0737423003838087)
        p.density.value = 2
        assert_almost_equal(p.sld.value, 1.7800363093423253)

    def test_from_pars(self):
        p = MaterialDensity('Co', 8.9, 'Cobalt')
        assert p.density.value == 8.9
        assert_almost_equal(p.sld.value,2.264541463379026)
        assert p.chemical_structure == 'Co'

    def test_chemical_structure_change(self):
        p = MaterialDensity('Co', 8.9, 'Cobalt')
        assert p.density.value == 8.9
        assert_almost_equal(p.sld.value, 2.264541463379026)
        assert_almost_equal(p.isld.value, 0.0)
        assert p.chemical_structure == 'Co'
        p.chemical_structure = 'B'
        assert p.density.value == 8.9
        assert_almost_equal(p.sld.value, 4.82010833570636)
        assert_almost_equal(p.isld.value, -0.19098540517806603)
        assert p.chemical_structure == 'B'

    def test_dict_repr(self):
        p = MaterialDensity()
        print(p._dict_repr)
        assert p._dict_repr == {
            'EasyMaterialDensity': {'sld': '2.074e-6 kmol/m^5', 'isld': '0.000e-6 kmol/m^5'},
            'chemical_structure': 'Si',
            'density': '2.33e+00 kg/L',
        }

    def test_dict_round_trip(self):
        p = MaterialDensity()
        p_dict = p.as_dict()
        global_object.map._clear()

        q = MaterialDensity.from_dict(p_dict)

        assert sorted(p.as_dict()) == sorted(q.as_dict())
