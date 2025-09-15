"""
Tests for Model class.
"""

__author__ = 'github.com/arm61'
__version__ = '0.0.1'

import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerCollection
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample import SurfactantLayer


class TestModel(unittest.TestCase):
    def test_default(self):
        p = Model()
        assert_equal(p.name, 'Model')
        assert_equal(p.interface, None)
        assert_equal(p.sample.name, 'EasySample')
        assert_equal(p.scale.display_name, 'scale')
        assert_equal(str(p.scale.unit), 'dimensionless')
        assert_equal(p.scale.value, 1.0)
        assert_equal(p.scale.min, 0.0)
        assert_equal(p.scale.max, np.inf)
        assert_equal(p.scale.fixed, True)
        assert_equal(p.background.display_name, 'background')
        assert_equal(str(p.background.unit), 'dimensionless')
        assert_equal(p.background.value, 1.0e-8)
        assert_equal(p.background.min, 0.0)
        assert_equal(p.background.max, np.inf)
        assert_equal(p.background.fixed, True)
        assert p._resolution_function.smearing([1]) == 5.0
        assert p._resolution_function.smearing([100]) == 5.0

    def test_from_pars(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, o2, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(
            sample=d,
            scale=2,
            background=1e-5,
            resolution_function=resolution_function,
            name='newModel',
        )
        assert_equal(mod.name, 'newModel')
        assert_equal(mod.interface, None)
        assert_equal(mod.sample.name, 'myModel')
        assert_equal(mod.scale.display_name, 'scale')
        assert_equal(str(mod.scale.unit), 'dimensionless')
        assert_equal(mod.scale.value, 2.0)
        assert_equal(mod.scale.min, 0.0)
        assert_equal(mod.scale.max, np.inf)
        assert_equal(mod.scale.fixed, True)
        assert_equal(mod.background.display_name, 'background')
        assert_equal(str(mod.background.unit), 'dimensionless')
        assert_equal(mod.background.value, 1.0e-5)
        assert_equal(mod.background.min, 0.0)
        assert_equal(mod.background.max, np.inf)
        assert_equal(mod.background.fixed, True)
        assert mod._resolution_function.smearing([1]) == 2.0
        assert mod._resolution_function.smearing([100]) == 2.0

    def test_add_assemblies(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        surfactant = SurfactantLayer()
        multilayer = Multilayer()
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel')
        assert_equal(len(mod.sample), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.sample), 2)
        assert_equal(mod.sample[1].name, 'oneLayerItem2')
        assert_equal(issubclass(mod.sample[1].__class__, RepeatingMultilayer), True)
        mod.add_assemblies(surfactant)
        assert_equal(len(mod.sample), 3)
        mod.add_assemblies(multilayer)
        assert_equal(len(mod.sample), 4)

    def test_add_assemblies_exception(self):
        # When
        mod = Model()

        # Then Expect
        with pytest.raises(ValueError):
            mod.add_assemblies('not an assembly')

    def test_add_assemblies_with_interface_refnx(self):
        interface = CalculatorFactory()
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_add_assemblies_with_interface_refl1d(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    # def test_add_assemblies_with_interface_bornagain(self):
    #     interface = CalculatorFactory()
    #     interface.switch('BornAgain')
    #     m1 = Material.from_pars(6.908, 0.278, 'Boron')
    #     m2 = Material.from_pars(0.487, 0.000, 'Potassium')
    #     l1 = Layer.from_pars(m1, 5.0, 2.0, 'thinBoron')
    #     l2 = Layer.from_pars(m2, 50.0, 1.0, 'thickPotassium')
    #     ls1 = Layers.from_pars(l1, l2, name='twoLayer1')
    #     ls2 = Layers.from_pars(l2, l1, name='twoLayer2')
    #     o1 = RepeatingMultilayer.from_pars(ls1, 2.0, 'twoLayerItem1')
    #     o2 = RepeatingMultilayer.from_pars(ls2, 1.0, 'oneLayerItem2')
    #     d = Sample.from_pars(o1, name='myModel')
    #     mod = Model(d, 2, 1e-5, 2.0, 'newModel', interface=interface)
    #     assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
    #     assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
    #     mod.add_assemblies(o2)
    #     assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
    #     assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_duplicate_assembly(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel')
        assert_equal(len(mod.sample), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.sample), 2)
        mod.duplicate_assembly(1)
        assert_equal(len(mod.sample), 3)
        assert_equal(mod.sample[2].name, 'oneLayerItem2 duplicate')
        assert_equal(issubclass(mod.sample[2].__class__, RepeatingMultilayer), True)

    def test_duplicate_assembly_with_interface_refnx(self):
        interface = CalculatorFactory()
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        mod.duplicate_assembly(1)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 3)

    def test_duplicate_assembly_with_interface_refl1d(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        mod.duplicate_assembly(1)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 3)

    # def test_duplicate_item_with_interface_bornagain(self):
    #     interface = CalculatorFactory()
    #     interface.switch('BornAgain')
    #     m1 = Material.from_pars(6.908, 0.278, 'Boron')
    #     m2 = Material.from_pars(0.487, 0.000, 'Potassium')
    #     l1 = Layer.from_pars(m1, 5.0, 2.0, 'thinBoron')
    #     l2 = Layer.from_pars(m2, 50.0, 1.0, 'thickPotassium')
    #     ls1 = Layers.from_pars(l1, l2, name='twoLayer1')
    #     ls2 = Layers.from_pars(l2, l1, name='twoLayer2')
    #     o1 = RepeatingMultilayer.from_pars(ls1, 2.0, 'twoLayerItem1')
    #     o2 = RepeatingMultilayer.from_pars(ls2, 1.0, 'oneLayerItem2')
    #     d = Sample.from_pars(o1, name='myModel')
    #     mod = Model(d, 2, 1e-5, 2.0, 'newModel', interface=interface)
    #     assert_equal(len(mod.interface()._wrapper.storage['assembly']), 1)
    #     mod.add_assemblies(o2)
    #     assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
    #     mod.duplicate_assembly(1)
    #     assert_equal(len(mod.interface()._wrapper.storage['item']), 3)

    def test_remove_assembly(self):
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel')
        assert_equal(len(mod.sample), 1)
        mod.add_assemblies(o2)
        assert_equal(len(mod.sample), 2)
        mod.remove_assembly(0)
        assert_equal(len(mod.sample), 1)

    def test_remove_assembly_with_interface_refnx(self):
        interface = CalculatorFactory()
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.remove_assembly(0)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_remove_assembly_with_interface_refl1d(self):
        interface = CalculatorFactory()
        interface.switch('refl1d')
        m1 = Material(6.908, -0.278, 'Boron')
        m2 = Material(0.487, 0.000, 'Potassium')
        l1 = Layer(m1, 5.0, 2.0, 'thinBoron')
        l2 = Layer(m2, 50.0, 1.0, 'thickPotassium')
        ls1 = LayerCollection(l1, l2, name='twoLayer1')
        ls2 = LayerCollection(l2, l1, name='twoLayer2')
        o1 = RepeatingMultilayer(ls1, 2.0, 'twoLayerItem1')
        o2 = RepeatingMultilayer(ls2, 1.0, 'oneLayerItem2')
        d = Sample(o1, name='myModel')
        resolution_function = PercentageFwhm(2.0)
        mod = Model(d, 2, 1e-5, resolution_function, 'newModel', interface=interface)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.add_assemblies(o2)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
        mod.remove_assembly(0)
        assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
        assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    # def test_remove_assembly_with_interface_bornagain(self):
    #     interface = CalculatorFactory()
    #     interface.switch('BornAgain')
    #     m1 = Material.from_pars(6.908, 0.278, 'Boron')
    #     m2 = Material.from_pars(0.487, 0.000, 'Potassium')
    #     l1 = Layer.from_pars(m1, 5.0, 2.0, 'thinBoron')
    #     l2 = Layer.from_pars(m2, 50.0, 1.0, 'thickPotassium')
    #     ls1 = Layers.from_pars(l1, l2, name='twoLayer1')
    #     ls2 = Layers.from_pars(l2, l1, name='twoLayer2')
    #     o1 = RepeatingMultilayer.from_pars(ls1, 2.0, 'twoLayerItem1')
    #     o2 = RepeatingMultilayer.from_pars(ls2, 1.0, 'oneLayerItem2')
    #     d = Sample.from_pars(o1, name='myModel')
    #     mod = Model(d, 2, 1e-5, 2.0, 'newModel', interface=interface)
    #     assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
    #     assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
    #     mod.add_assemblies(o2)
    #     assert_equal(len(mod.interface()._wrapper.storage['item']), 2)
    #     assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)
    #     mod.remove_assembly(0)
    #     assert_equal(len(mod.interface()._wrapper.storage['item']), 1)
    #     assert_equal(len(mod.interface()._wrapper.storage['layer']), 2)

    def test_remove_all_assemblies(self):
        # when
        mod = Model()

        # Then
        mod.remove_assembly(0)
        mod.remove_assembly(0)

        # Expect
        assert_equal(len(mod.sample), 0)

    def test_resolution_function(self):
        mock_resolution_function = MagicMock()
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model = Model(interface=interface)

        # Then
        model.resolution_function = mock_resolution_function

        # Expect
        assert model.resolution_function == mock_resolution_function

    def test_resolution_function_interface_refl1d(self):
        mock_resolution_function = MagicMock()
        interface = CalculatorFactory()
        interface.switch('refl1d')
        model = Model(interface=interface)

        # Then
        model.resolution_function = mock_resolution_function

        # Expect
        assert model.interface()._wrapper._resolution_function == mock_resolution_function

    def test_set_resolution_function_interface_refnx(self):
        mock_resolution_function = MagicMock()
        interface = CalculatorFactory()
        interface.switch('refnx')
        model = Model(interface=interface)

        # Then
        model.resolution_function = mock_resolution_function

        # Expect
        assert model.interface()._wrapper._resolution_function == mock_resolution_function

    def test_repr(self):
        model = Model()

        assert (
            model.__repr__()
            == "Model:\n  scale: 1.0\n  background: 1.0e-08\n  resolution: 5.0 %\n  color: '#0173B2'\n  sample:\n    EasySample:\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n"  # noqa: E501
        )

    def test_repr_resolution_function(self):
        resolution_function = LinearSpline([0, 10], [0, 10])
        model = Model()
        model.resolution_function = resolution_function
        assert (
            model.__repr__()
            == "Model:\n  scale: 1.0\n  background: 1.0e-08\n  resolution: function of Q\n  color: '#0173B2'\n  sample:\n    EasySample:\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n    - EasyMultilayer:\n        EasyLayerCollection:\n        - EasyLayer:\n            material:\n              EasyMaterial:\n                sld: 4.186e-6 1/Å^2\n                isld: 0.000e-6 1/Å^2\n            thickness: 10.000 Å\n            roughness: 3.300 Å\n"  # noqa: E501
        )


@pytest.mark.parametrize(
    'interface',
    [None, CalculatorFactory()],
)
def test_dict_round_trip(interface):
    # When
    resolution_function = LinearSpline([0, 10], [0, 10])
    model = Model(interface=interface)
    model.resolution_function = resolution_function
    for additional_layer in [SurfactantLayer(), Multilayer(), RepeatingMultilayer()]:
        model.add_assemblies(additional_layer)
    src_dict = model.as_dict()
    global_object.map._clear()

    # Then
    model_from_dict = Model.from_dict(src_dict)

    # Expect
    assert sorted(model.as_dict(skip=['resolution_function', 'interface'])) == sorted(
        model_from_dict.as_dict(skip=['resolution_function', 'interface'])
    )
    assert model._resolution_function.smearing(5.5) == model_from_dict._resolution_function.smearing(5.5)
    if interface is not None:
        assert model.interface().name == model_from_dict.interface().name
        assert_almost_equal(
            model.interface().reflectivity_profile([0.3], model.unique_name),
            model_from_dict.interface().reflectivity_profile([0.3], model_from_dict.unique_name),
        )
