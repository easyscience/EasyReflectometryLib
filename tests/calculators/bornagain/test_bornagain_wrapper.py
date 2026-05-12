# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the BornAgain wrapper.
"""


# import unittest
# import numpy as np
# from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
# from easyreflectometry.calculators.bornagain import BornAgain
# import bornagain as ba

# class TestBornAgain(unittest.TestCase):
#     def test_init(self):
#         p = BornAgain()
#         assert_equal(list(p.storage.keys()), ['material', 'layer', 'layer_material', 'roughness', 'item', 'item_repeats',
#                                               'model', 'model_items', 'model_parameters'])
#         assert_equal(issubclass(p.storage['material'].__class__, dict), True)

#     def test_create_material(self):
#         p = BornAgain()
#         p.create_material('Si')
#         assert_equal(list(p.storage['material'].keys()), ['Si'])
#         assert_almost_equal(p.storage['material']['Si'].materialData().real, 0.0)
#         assert_almost_equal(p.storage['material']['Si'].materialData().imag, 0.0)
#         assert_equal(p.storage['material']['Si'].getName(), 'Si')

#     def test_update_material(self):
#         p = BornAgain()
#         p.create_material('B')
#         p.update_material('B', real=6.908, imag=0.278)
#         assert_equal(list(p.storage['material'].keys()), ['B'])
#         assert_almost_equal(p.storage['material']
#                             ['B'].materialData().real, 6.908e-6)
#         assert_almost_equal(p.storage['material']
#                             ['B'].materialData().imag, 0.278e-6)

#     def test_update_material_neg_imag(self):
#         p = BornAgain()
#         p.create_material('B')
#         with self.assertRaises(ValueError):
#             p.update_material('B', real=6.908, imag=-0.278)

#     def test_get_material_value(self):
#         p = BornAgain()
#         p.create_material('B')
#         p.update_material('B', real=6.908, imag=0.278)
#         assert_equal(list(p.storage['material'].keys()), ['B'])
#         assert_almost_equal(p.get_material_value('B', 'real'), 6.908)
#         assert_almost_equal(p.get_material_value('B', 'imag'), 0.278)

#     def test_create_layer(self):
#         p = BornAgain()
#         p.create_layer('Si')
#         assert_equal(list(p.storage['layer'].keys()), ['Si'])
#         assert_almost_equal(p.storage['layer']['Si'].thickness(), 0)
#         assert_almost_equal(p.storage['roughness']['Si'].getSigma(), 0)

#     def test_update_layer(self):
#         p = BornAgain()
#         p.create_material('Si')
#         p.create_layer('Si')
#         p.assign_material_to_layer('Si', 'Si')
#         p.update_layer('Si', thickness=10, sigma=5)
#         assert_almost_equal(p.storage['layer']['Si'].thickness(), 1)
#         assert_almost_equal(p.storage['roughness']['Si'].getSigma(), 0.5)

#     def test_get_layer_value(self):
#         p = BornAgain()
#         p.create_material('Si')
#         p.create_layer('Si')
#         p.assign_material_to_layer('Si', 'Si')
#         p.update_layer('Si', thickness=10, sigma=5)
#         assert_almost_equal(p.get_layer_value('Si', 'thickness'), 10)
#         assert_almost_equal(p.get_layer_value('Si', 'sigma'), 5)

#     def test_create_item(self):
#         p = BornAgain()
#         p.create_item('SiNi')
#         assert_equal(list(p.storage['item'].keys()), ['SiNi'])
#         assert_equal(p.storage['item']['SiNi'], [])
#         assert_equal(p.storage['item_repeats']['SiNi'], 1)

#     def test_update_item(self):
#         p = BornAgain()
#         p.create_item('SiNi')
#         p.update_item('SiNi', repeats=10)
#         assert_almost_equal(p.storage['item_repeats']['SiNi'], 10)

#     def test_get_item_value(self):
#         p = BornAgain()
#         p.create_item('SiNi')
#         p.update_item('SiNi', repeats=10)
#         assert_almost_equal(p.get_item_value('SiNi', 'repeats'), 10)

#     def test_create_model(self):
#         p = BornAgain()
#         p.create_model()
#         assert_equal(isinstance(p.storage['model'], ba.Multilayer), True)
#         assert_equal(p.storage['model'].roughnessModel(), 2)
#         assert_equal(list(p.storage['model_parameters'].keys()), ['scale', 'background', 'resolution'])

#     def test_update_model(self):
#         p = BornAgain()
#         p.create_model()
#         p.update_model('model', scale=2, background=1e-3, resolution=2.0)
#         assert_almost_equal(p.storage['model_parameters']['scale'], 2)
#         assert_almost_equal(p.storage['model_parameters']['background'], 1e-3)
#         assert_almost_equal(p.storage['model_parameters']['resolution'], 2.0)

#     def test_get_model_value(self):
#         p = BornAgain()
#         p.create_model()
#         p.update_model('model', scale=2, background=1e-3, resolution=2.0)
#         assert_almost_equal(p.get_model_value('model', 'scale'), 2)
#         assert_almost_equal(p.get_model_value('model', 'background'), 1e-3)
#         assert_almost_equal(p.get_model_value('model', 'resolution'), 2.0)

#     def test_assign_material_to_layer(self):
#         p = BornAgain()
#         p.create_material('B')
#         p.update_material('B', real=6.908, imag=0.278)
#         p.create_layer('B_layer')
#         p.assign_material_to_layer('B', 'B_layer')
#         assert_almost_equal(
#             p.storage['material'][p.storage['layer_material']['B_layer']].materialData().real, 6.908e-6)
#         assert_almost_equal(
#             p.storage['material'][p.storage['layer_material']['B_layer']].materialData().imag, 0.278e-6)

#     def test_add_layer_to_item(self):
#         p = BornAgain()
#         p.create_material('B')
#         p.update_material('B', real=6.908, imag=0.278)
#         p.create_layer('B_layer')
#         p.update_layer('B_layer', sigma=10)
#         p.assign_material_to_layer('B', 'B_layer')
#         p.create_item('B_item')
#         assert_equal(len(p.storage['item']['B_item']), 0)
#         p.add_layer_to_item('B_layer', 'B_item')
#         assert_equal(len(p.storage['item']['B_item']), 1)

#     def test_add_item(self):
#         p = BornAgain()
#         p.create_material('B')
#         p.update_material('B', real=6.908, imag=0.278)
#         p.create_layer('B_layer')
#         p.assign_material_to_layer('B', 'B_layer')
#         p.create_item('B_item')
#         p.add_layer_to_item('B_layer', 'B_item')
#         p.create_model()
#         assert_equal(len(p.storage['model_items']), 0)
#         p.add_item('B_item')
#         assert_equal(len(p.storage['model_items']), 1)

#     def test_remove_layer_from_item(self):
#         p = BornAgain()
#         p.create_material('B')
#         p.update_material('B', real=6.908, imag=0.278)
#         p.create_layer('B_layer')
#         p.assign_material_to_layer('B', 'B_layer')
#         p.create_item('B_item')
#         p.add_layer_to_item('B_layer', 'B_item')
#         assert_equal(len(p.storage['item']['B_item']), 1)
#         p.remove_layer_from_item('B_layer', 'B_item')
#         assert_equal(len(p.storage['item']['B_item']), 0)

#     def test_remove_item(self):
#         p = BornAgain()
#         p.create_material('B')
#         p.update_material('B', real=6.908, imag=0.278)
#         p.create_layer('B_layer')
#         p.assign_material_to_layer('B', 'B_layer')
#         p.create_item('B_item')
#         p.add_layer_to_item('B_layer', 'B_item')
#         p.create_model()
#         p.add_item('B_item')
#         assert_equal(len(p.storage['model_items']), 1)
#         p.remove_item('B_item')
#         assert_equal(len(p.storage['model_items']), 0)

#     def test_calculate(self):
#         p = BornAgain()
#         p.create_material('Material1')
#         p.update_material('Material1', real=0.000, imag=0.000)
#         p.create_material('Material2')
#         p.update_material('Material2', real=2.000, imag=0.000)
#         p.create_material('Material3')
#         p.update_material('Material3', real=4.000, imag=0.000)
#         p.create_model()
#         p.create_layer('Layer1')
#         p.assign_material_to_layer('Material1', 'Layer1')
#         p.create_layer('Layer2')
#         p.assign_material_to_layer('Material2', 'Layer2')
#         p.update_layer('Layer2', thickness=10, sigma=1.0)
#         p.create_layer('Layer3')
#         p.assign_material_to_layer('Material3', 'Layer3')
#         p.update_layer('Layer3', sigma=1.0)
#         p.create_item('Item')
#         p.add_layer_to_item('Layer1', 'Item')
#         p.add_layer_to_item('Layer2', 'Item')
#         p.add_layer_to_item('Layer3', 'Item')
#         p.add_item('Item')
#         p.update_model('model', background=1e-7, resolution=5)
#         q = np.linspace(0.001, 0.3, 10)
#         expected = [
#             9.99956517e-01, 2.16286891e-03, 1.14086254e-04, 1.93031759e-05,
#             4.94188894e-06, 1.54191953e-06, 5.45592112e-07, 2.26619392e-07,
#             1.26726993e-07, 1.01842852e-07
#         ]
#         assert_allclose(p.calculate(q), expected, rtol=0.04)

#     def test_calculate2(self):
#         p = BornAgain()
#         p.create_material('Material1')
#         p.update_material('Material1', real=0.000, imag=0.000)
#         p.create_material('Material2')
#         p.update_material('Material2', real=2.000, imag=0.000)
#         p.create_material('Material3')
#         p.update_material('Material3', real=4.000, imag=0.000)
#         p.create_layer('Layer1')
#         p.assign_material_to_layer('Material1', 'Layer1')
#         p.create_layer('Layer2')
#         p.assign_material_to_layer('Material2', 'Layer2')
#         p.update_layer('Layer2', thickness=10, sigma=1.0)
#         p.create_layer('Layer3')
#         p.assign_material_to_layer('Material3', 'Layer3')
#         p.update_layer('Layer3', sigma=1.0)
#         p.create_item('Item1')
#         p.add_layer_to_item('Layer1', 'Item1')
#         p.create_item('Item2')
#         p.add_layer_to_item('Layer2', 'Item2')
#         p.add_layer_to_item('Layer1', 'Item2')
#         p.create_item('Item3')
#         p.add_layer_to_item('Layer3', 'Item3')
#         p.create_model()
#         p.add_item('Item1')
#         p.add_item('Item2')
#         p.add_item('Item3')
#         p.update_item('Item2', repeats=10)
#         p.update_model('model', background=1e-7, resolution=5)
#         q = np.linspace(0.001, 0.3, 10)
#         expected = [1.000000e+00, 1.814452e-05, 1.225890e-04, 2.454331e-06,
#                     6.676318e-06, 8.362728e-07, 1.141096e-06, 4.090968e-07,
#                     3.489857e-07, 2.470789e-07]
#         assert_allclose(p.calculate(q), expected, rtol=0.01)
#         assert_allclose(p.calculate(q), expected, rtol=0.01)
