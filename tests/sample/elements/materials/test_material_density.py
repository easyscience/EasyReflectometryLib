# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

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
        assert_almost_equal(p.sld.value, 2.264541463379026)
        assert p.chemical_structure == 'Co'

    def test_chemical_structure_change(self):
        p = MaterialDensity('Co', 8.9, 'Cobalt')
        assert p.density.value == 8.9
        assert_almost_equal(p.sld.value, 2.264541463379026)
        assert_almost_equal(p.isld.value, 0.0)
        assert p.chemical_structure == 'Co'
        p.chemical_structure = 'B'
        assert p.density.value == 8.9
        # Changing the structure must also refresh the molar mass (issue #369);
        # the SLD is computed with boron's b AND boron's M (10.81 g/mol), not
        # the stale cobalt molar mass.
        assert_almost_equal(p.molecular_weight.value, 10.81)
        assert_almost_equal(p.sld.value, 26.277925961998147)
        assert_almost_equal(p.isld.value, -1.0412008400037)
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

    def test_density_mutation_propagates_after_round_trip(self):
        """Regression: after ``from_dict`` reattaches the saved ``_density``
        Parameter, mutating it must propagate to ``sld`` / ``isld`` (which
        are constrained off it). The ``__init__``-time constraint references
        the temporary constructor Parameter; ``from_dict`` rebuilds the
        graph so subsequent mutations propagate correctly.
        """
        p = MaterialDensity(chemical_structure='Si', density=2.33)
        original_sld = p.sld.value
        p_dict = p.as_dict()
        global_object.map._clear()

        q = MaterialDensity.from_dict(p_dict)
        assert_almost_equal(q.sld.value, original_sld)

        q.density = 4.66
        # SLD scales linearly with density (constraint: d * sl / mw, etc.)
        assert_almost_equal(q.sld.value, 2 * original_sld)
