"""
Tests for LayerCollection class.
"""

__author__ = 'github.com/arm61'
__version__ = '0.0.1'

import unittest

from easyscience import global_object
from numpy.testing import assert_equal

from easyreflectometry.sample.assemblies.repeating_multilayer import RepeatingMultilayer
from easyreflectometry.sample.collections.layer_collection import LayerCollection
from easyreflectometry.sample.elements.layers.layer import Layer
from easyreflectometry.sample.elements.materials.material import Material


class TestLayerCollection(unittest.TestCase):
    def test_default(self):
        p = LayerCollection()
        assert_equal(p.name, 'EasyLayerCollection')
        assert_equal(p.interface, None)
        assert_equal(len(p), 0)

    def test_from_pars(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        layers = LayerCollection(p, q, name='twoLayer')
        assert_equal(layers.name, 'twoLayer')
        assert_equal(layers.interface, None)
        assert_equal(len(layers), 2)
        assert_equal(layers[0].name, 'thinBoron')
        assert_equal(layers[1].name, 'thickPotassium')

    def test_from_pars_item(self):
        m = Material(6.908, -0.278, 'Boron')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        i = RepeatingMultilayer(LayerCollection(), 2)
        layers = LayerCollection(p, i, name='twoLayer')
        assert_equal(layers.name, 'twoLayer')
        assert_equal(layers.interface, None)

    def test_dict_repr(self):
        p = LayerCollection(layers=[Layer(), Layer()])
        assert p._dict_repr == {
            'EasyLayerCollection': [
                {
                    'EasyLayer': {
                        'material': {'EasyMaterial': {'sld': '4.186e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
                        'thickness': '10.000 Å',
                        'roughness': '3.300 Å',
                    }
                },
                {
                    'EasyLayer': {
                        'material': {'EasyMaterial': {'sld': '4.186e-6 1/Å^2', 'isld': '0.000e-6 1/Å^2'}},
                        'thickness': '10.000 Å',
                        'roughness': '3.300 Å',
                    }
                },
            ]
        }

    def test_repr(self):
        p = LayerCollection([Layer(), Layer()])
        assert (
            p.__repr__()
            == 'EasyLayerCollection:\n- EasyLayer:\n    material:\n      EasyMaterial:\n        sld: 4.186e-6 1/Å^2\n        isld: 0.000e-6 1/Å^2\n    thickness: 10.000 Å\n    roughness: 3.300 Å\n- EasyLayer:\n    material:\n      EasyMaterial:\n        sld: 4.186e-6 1/Å^2\n        isld: 0.000e-6 1/Å^2\n    thickness: 10.000 Å\n    roughness: 3.300 Å\n'  # noqa: E501
        )

    def test_dict_round_trip(self):
        # When
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        r = LayerCollection()
        r.insert(0, p)
        r.append(q)
        r_dict = r.as_dict()
        global_object.map._clear()

        # Then
        s = LayerCollection.from_dict(r_dict)

        # Expect
        assert sorted(r.as_dict()) == sorted(s.as_dict())

    def test_add_layer(self):
        # When
        p = LayerCollection()
        m = Layer(name='Layer m')

        # Then
        p.add_layer()
        p.add_layer(m)

        # Expect
        assert p[1] == m

    def test_duplicate_layer(self):
        # When
        p = LayerCollection()
        m = Layer(name='Layer m')
        p.add_layer(m)

        # Then
        p.duplicate_layer(0)

        # Expect
        assert p[1].name == 'Layer m duplicate'
