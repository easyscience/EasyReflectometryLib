"""
Tests for MultiLayer class module
"""

__author__ = 'github.com/arm61'
__version__ = '0.0.1'

import unittest

from easyscience import global_object
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from easyreflectometry.calculators.factory import CalculatorFactory
from easyreflectometry.sample.assemblies.multilayer import Multilayer
from easyreflectometry.sample.collections.layer_collection import LayerCollection
from easyreflectometry.sample.elements.layers.layer import Layer
from easyreflectometry.sample.elements.materials.material import Material


class TestMultilayer(unittest.TestCase):
    def test_default(self):
        p = Multilayer()
        assert_equal(p.name, 'EasyMultilayer')
        assert_equal(p._type, 'Multi-layer')
        assert_equal(p.interface, None)
        assert_equal(len(p.layers), 1)
        assert_equal(p.layers.name, 'EasyLayerCollection')

    def test_default_empty(self):
        p = Multilayer(populate_if_none=False)
        assert_equal(p.name, 'EasyMultilayer')
        assert_equal(p._type, 'Multi-layer')
        assert_equal(p.interface, None)
        assert_equal(len(p.layers), 0)

    def test_from_pars(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        layers = LayerCollection(p, q, name='twoLayer')
        o = Multilayer(layers, 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o._type, 'Multi-layer')
        assert_equal(o.interface, None)
        assert_equal(o.layers.name, 'twoLayer')

    def test_from_pars_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        o = Multilayer(p, 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o.interface, None)
        assert_equal(o.layers.name, 'thinBoron')

    def test_from_pars_layer_list(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 15.0, 2.0, 'layerPotassium')
        o = Multilayer([p, q], 'twoLayerItem')
        assert_equal(o.name, 'twoLayerItem')
        assert_equal(o.interface, None)
        assert_equal(o.layers.name, 'thinBoron/layerPotassium')

    def test_add_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = Multilayer(p, 'twoLayerItem')
        assert_equal(len(o.layers), 1)
        o.add_layer(q)
        assert_equal(len(o.layers), 2)
        assert_equal(o.layers[1].name, 'thickPotassium')

    def test_add_layer_with_interface_refnx(self):
        interface = CalculatorFactory()
        interface.switch('refnx')
        m = Material(6.908, -0.278, 'Boron', interface=interface)
        k = Material(0.487, 0.000, 'Potassium', interface=interface)
        p = Layer(m, 5.0, 2.0, 'thinBoron', interface=interface)
        q = Layer(k, 50.0, 1.0, 'thickPotassium', interface=interface)
        o = Multilayer(p, 'twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.interface()._wrapper.storage['item'][o.unique_name].components[1].thick.value, 50.0)

    def test_duplicate_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = Multilayer(p, 'twoLayerItem')
        assert_equal(len(o.layers), 1)
        o.add_layer(q)
        assert_equal(len(o.layers), 2)
        o.duplicate_layer(1)
        assert_equal(len(o.layers), 3)
        assert_equal(o.layers[1].name, 'thickPotassium')
        assert_equal(o.layers[2].name, 'thickPotassium duplicate')

    def test_duplicate_layer_with_interface_refnx(self):
        interface = CalculatorFactory()
        interface.switch('refnx')
        m = Material(6.908, -0.278, 'Boron', interface=interface)
        k = Material(0.487, 0.000, 'Potassium', interface=interface)
        p = Layer(m, 5.0, 2.0, 'thinBoron', interface=interface)
        q = Layer(k, 50.0, 1.0, 'thickPotassium', interface=interface)
        o = Multilayer(p, 'twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.interface()._wrapper.storage['item'][o.unique_name].components[1].thick.value, 50.0)
        o.duplicate_layer(1)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 3)
        assert_equal(o.interface()._wrapper.storage['item'][o.unique_name].components[2].thick.value, 50.0)
        assert_raises(
            AssertionError,
            assert_equal,
            o.interface()._wrapper.storage['item'][o.unique_name].components[1].name,
            o.interface()._wrapper.storage['item'][o.unique_name].components[2].name,
        )

    def test_remove_layer(self):
        m = Material(6.908, -0.278, 'Boron')
        k = Material(0.487, 0.000, 'Potassium')
        p = Layer(m, 5.0, 2.0, 'thinBoron')
        q = Layer(k, 50.0, 1.0, 'thickPotassium')
        o = Multilayer(p, 'twoLayerItem')
        assert_equal(len(o.layers), 1)
        o.add_layer(q)
        assert_equal(len(o.layers), 2)
        assert_equal(o.layers[1].name, 'thickPotassium')
        o.remove_layer(1)
        assert_equal(len(o.layers), 1)
        assert_equal(o.layers[0].name, 'thinBoron')

    def test_remove_layer_with_interface_refnx(self):
        interface = CalculatorFactory()
        interface.switch('refnx')
        m = Material(6.908, -0.278, 'Boron', interface=interface)
        k = Material(0.487, 0.000, 'Potassium', interface=interface)
        p = Layer(m, 5.0, 2.0, 'thinBoron', interface=interface)
        q = Layer(k, 50.0, 1.0, 'thickPotassium', interface=interface)
        o = Multilayer(p, name='twoLayerItem', interface=interface)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        o.add_layer(q)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 2)
        assert_equal(o.layers[1].name, 'thickPotassium')
        o.remove_layer(1)
        assert_equal(len(o.interface()._wrapper.storage['item'][o.unique_name].components), 1)
        assert_equal(o.layers[0].name, 'thinBoron')

    def test_repr(self):
        p = Multilayer()
        assert (
            p.__repr__()
            == 'EasyMultilayer:\n  EasyLayerCollection:\n  - EasyLayer:\n      material:\n        EasyMaterial:\n          sld: 4.186e-6 1/Å^2\n          isld: 0.000e-6 1/Å^2\n      thickness: 10.000 Å\n      roughness: 3.300 Å\n'  # noqa: E501
        )

    def test_dict_round_trip(self):
        p = Multilayer()
        p_dict = p.as_dict()
        global_object.map._clear()

        q = Multilayer.from_dict(p_dict)
        assert sorted(p.as_dict()) == sorted(q.as_dict())
