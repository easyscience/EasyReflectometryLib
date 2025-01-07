"""
Tests for LayerAreaPerMolecule class.
"""

import unittest

from easyscience import global_object
from numpy.testing import assert_almost_equal

from easyreflectometry.sample.elements.layers.layer_area_per_molecule import LayerAreaPerMolecule
from easyreflectometry.sample.elements.materials.material import Material


class TestLayerAreaPerMolecule(unittest.TestCase):
    def test_default(self):
        p = LayerAreaPerMolecule()
        assert p.molecular_formula == 'C10H18NO8P'
        assert p.area_per_molecule == 48.2
        assert str(p._area_per_molecule.unit) == 'Å^2'
        assert p._area_per_molecule.fixed is True
        assert p.thickness.value == 10.0
        assert str(p.thickness.unit) == 'Å'
        assert p.thickness.fixed is True
        assert p.roughness.value == 3.3
        assert str(p.roughness.unit) == 'Å'
        assert p.roughness.fixed is True
        assert_almost_equal(p.material.sld, 2.268770124481328)
        assert_almost_equal(p.material.isld, 0)
        assert p.material.name == 'C10H18NO8P in D2O'
        assert p.solvent.sld.value == 6.36
        assert p.solvent.isld.value == 0
        assert p.solvent.name == 'D2O'
        assert p.solvent_fraction == 0.2
        assert str(p.material._fraction.unit) == 'dimensionless'
        assert p.material._fraction.fixed is True

    def test_from_pars(self):
        h2o = Material(-0.561, 0, 'H2O')
        p = LayerAreaPerMolecule(
            molecular_formula='C8O10H12P',
            thickness=12,
            solvent=h2o,
            solvent_fraction=0.5,
            area_per_molecule=50,
            roughness=2,
            name='PG/H2O',
        )
        assert p.molecular_formula == 'C8O10H12P'
        assert p.area_per_molecule == 50
        assert p.thickness.value == 12
        assert p.roughness.value == 2
        assert p.solvent.sld.value == -0.561
        assert p.solvent.isld.value == 0
        assert p.solvent_fraction == 0.5

    def test_from_pars_constraint(self):
        h2o = Material(-0.561, 0, 'H2O')
        p = LayerAreaPerMolecule(
            molecular_formula='C8O10H12P',
            thickness=12,
            solvent=h2o,
            solvent_fraction=0.5,
            area_per_molecule=50,
            roughness=2,
            name='PG/H2O',
        )
        assert p.molecular_formula == 'C8O10H12P'
        assert p.area_per_molecule == 50
        assert_almost_equal(p.material.sld, 0.31494833333333333)
        assert p.thickness.value == 12
        assert p.roughness.value == 2
        assert p.solvent.sld.value == -0.561
        assert p.solvent.isld.value == 0
        assert p.solvent_fraction == 0.5
        p.area_per_molecule = 30
        assert p.area_per_molecule == 30
        assert_almost_equal(p.material.sld, 0.7119138888888887)
        p.thickness.value = 10
        assert p.thickness.value == 10
        assert_almost_equal(p.material.sld, 0.9103966666666665)

    def test_solvent_change(self):
        h2o = Material(-0.561, 0, 'H2O')
        p = LayerAreaPerMolecule(
            molecular_formula='C8O10H12P',
            thickness=12,
            solvent=h2o,
            solvent_fraction=0.5,
            area_per_molecule=50,
            roughness=2,
            name='PG/H2O',
        )
        assert p.molecular_formula == 'C8O10H12P'
        assert p.area_per_molecule == 50
        print(p.material)
        assert_almost_equal(p.material.sld,  0.31494833333333333)
        assert p.thickness.value == 12
        assert p.roughness.value == 2
        assert p.solvent.sld.value == -0.561
        assert p.solvent.isld.value == 0
        assert p.solvent_fraction == 0.5
        d2o = Material(6.335, 0, 'D2O')
        p.solvent = d2o
        assert p.molecular_formula == 'C8O10H12P'
        assert p.area_per_molecule == 50
        assert_almost_equal(p.material.sld, 3.762948333333333)
        assert p.thickness.value == 12
        assert p.roughness.value == 2
        assert p.solvent.sld.value == 6.335
        assert p.solvent.isld.value == 0
        assert p.solvent_fraction == 0.5

    def test_molecular_formula_change(self):
        h2o = Material(-0.561, 0, 'H2O')
        p = LayerAreaPerMolecule(
            molecular_formula='C8O10H12P',
            thickness=12,
            solvent=h2o,
            solvent_fraction=0.5,
            area_per_molecule=50,
            roughness=2,
            name='PG/H2O',
        )
        assert p.molecular_formula == 'C8O10H12P'
        assert p.area_per_molecule == 50
        assert_almost_equal(p.material.sld, 0.31494833333333333)
        assert p.thickness.value == 12
        assert p.roughness.value == 2

        assert p.solvent.sld.value == -0.561
        assert p.solvent.isld.value == 0
        assert p.solvent_fraction == 0.5
        assert p.material.name == 'C8O10H12P in H2O'
        p.molecular_formula = 'C8O10D12P'
        assert p.molecular_formula == 'C8O10D12P'
        assert p.area_per_molecule == 50
        assert_almost_equal(p.material.sld, 1.3558483333333333)
        assert p.thickness.value == 12
        assert p.roughness.value == 2
        assert p.solvent.sld.value == -0.561
        assert p.solvent.isld.value == 0
        assert p.solvent_fraction == 0.5
        assert p.material.name == 'C8O10D12P in H2O'

    def test_dict_repr(self):
        p = LayerAreaPerMolecule()
        assert p._dict_repr == {
            'EasyLayerAreaPerMolecule': {
                'material': {
                    'C10H18NO8P in D2O': {
                        'solvent_fraction': '0.200 dimensionless',
                        'sld': '2.269e-6 1/Å^2',
                        'isld': '0.000e-6 1/Å^2',
                        'material': {'C10H18NO8P': {'sld': '1.246e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
                        'solvent': {'D2O': {'sld': '6.360e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
                    }
                },
                'thickness': '10.000 Å',
                'roughness': '3.300 Å',
            },
            'molecular_formula': 'C10H18NO8P',
            'area_per_molecule': '48.20 Å^2',
        }

    def test_dict_round_trip(self):
        # When
        solvent = Material(-0.561, 0, 'H2O')
        p = LayerAreaPerMolecule(
            molecular_formula='CO2',
            solvent=solvent,
            solvent_fraction=0.5,
            area_per_molecule=50,
            thickness=10,
            roughness=3,
        )
        p_dict = p.as_dict()
        global_object.map._clear()

        # Then
        q = LayerAreaPerMolecule.from_dict(p_dict)

        # Expect
        assert sorted(p.as_data_dict()) == sorted(q.as_data_dict())
