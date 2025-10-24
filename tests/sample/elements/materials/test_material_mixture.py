from unittest.mock import MagicMock

from easyscience import global_object
from numpy.testing import assert_almost_equal

from easyreflectometry.sample.elements.materials.material import Material
from easyreflectometry.sample.elements.materials.material_mixture import MaterialMixture


class TestMaterialMixture:
    def test_default(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        assert str(material_mixture._sld.unit) == '1/Å^2'
        assert str(material_mixture._isld.unit) == '1/Å^2'

    def test_default_constraint(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        material_mixture.material_a.sld.value = 0
        material_mixture.material_b.isld.value = -1
        assert_almost_equal(material_mixture.sld, 2.093)
        assert_almost_equal(material_mixture.isld, -0.5)
        assert str(material_mixture._sld.unit) == '1/Å^2'
        assert str(material_mixture._isld.unit) == '1/Å^2'

    def test_fraction_constraint(self):
        p = Material()
        q = Material(6.908, -0.278, 'Boron')
        material_mixture = MaterialMixture(p, q, 0.2)
        assert material_mixture.fraction == 0.2
        assert_almost_equal(material_mixture.sld, 4.7304)
        assert_almost_equal(material_mixture.isld, -0.0556)
        material_mixture._fraction.value = 0.5
        assert material_mixture.fraction == 0.5
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)

    def test_material_a_change(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        q = Material(6.908, -0.278, 'Boron')
        material_mixture.material_a = q
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)

    def test_material_b_change(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        q = Material(6.908, -0.278, 'Boron')
        material_mixture.material_b = q
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)

    def test_material_b_change_double(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.186)
        assert_almost_equal(material_mixture.isld, 0)
        q = Material(6.908, -0.278, 'Boron')
        material_mixture.material_b = q
        assert material_mixture.name == 'EasyMaterial/Boron'
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 5.54700)
        assert_almost_equal(material_mixture.isld, -0.1390)
        r = Material(0.00, 0.00, 'ACMW')
        material_mixture.material_b = r
        assert material_mixture.name == 'EasyMaterial/ACMW'
        assert material_mixture.fraction == 0.5
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 2.0930)
        assert_almost_equal(material_mixture.isld, 0.0000)

    def test_from_pars(self):
        p = Material()
        q = Material(6.908, -0.278, 'Boron')
        material_mixture = MaterialMixture(p, q, 0.2)
        assert material_mixture.fraction == 0.2
        assert str(material_mixture._fraction.unit) == 'dimensionless'
        assert_almost_equal(material_mixture.sld, 4.7304)
        assert_almost_equal(material_mixture.isld, -0.0556)
        assert str(material_mixture._sld.unit) == '1/Å^2'
        assert str(material_mixture._isld.unit) == '1/Å^2'

    def test_dict_repr(self) -> None:
        material_mixture = MaterialMixture()
        assert material_mixture._dict_repr == {
            'EasyMaterial/EasyMaterial': {
                'fraction': '0.500 dimensionless',
                'sld': '4.186e-6 1/Å^2',
                'isld': '0.000e-6 1/Å^2',
                'material_a': {'EasyMaterial': {'sld': '4.186e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
                'material_b': {'EasyMaterial': {'sld': '4.186e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
            }
        }

    def test_dict_round_trip(self) -> None:
        # When
        p = MaterialMixture()
        p_dict = p.as_dict()
        global_object.map._clear()

        # Then
        q = MaterialMixture.from_dict(p_dict)

        # Expect
        assert sorted(p.as_dict()) == sorted(q.as_dict())

    def test_update_name(self) -> None:
        # When
        material_mixture = MaterialMixture()
        mock_material_a = MagicMock()
        mock_material_a.name = 'name_a'
        material_mixture._material_a = mock_material_a
        mock_material_b = MagicMock()
        mock_material_b.name = 'name_b'
        material_mixture._material_b = mock_material_b

        # Then
        material_mixture._update_name()

        # Expect
        assert material_mixture.name == 'name_a/name_b'
