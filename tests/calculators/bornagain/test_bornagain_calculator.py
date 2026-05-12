# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for BornAgain calculator.
"""


# import os
# import unittest
# import numpy as np
# from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
# from easyreflectometry.interfaces.bornagain import BornAgain
# from easyreflectometry.sample.material import Material

# class TestBornAgain(unittest.TestCase):
#     def test_init(self):
#         p = BornAgain()
#         assert_equal(list(p.calculator.storage.keys()),
#                      ['material', 'layer', 'layer_material', 'roughness', 'item',
#                       'item_repeats', 'model', 'model_items', 'model_parameters'])
#         assert_equal(p._material_link['sld'], 'real')
#         assert_equal(p._material_link['isld'], 'imag')
#         assert_equal(p._layer_link['thickness'], 'thickness')
#         assert_equal(p._layer_link['roughness'], 'sigma')
#         assert_equal(p._item_link['repetitions'], 'repeats')
#         assert_equal(p._model_link['scale'], 'scale')
#         assert_equal(p._model_link['background'], 'background')
#         assert_equal(p._model_link['resolution'], 'resolution')
#         assert_equal(p.name, 'BornAgain')

#     def test_fit_func(self):
#         p = BornAgain()
#         p.calculator.create_material('Material1')
#         p.calculator.update_material('Material1', real=0.000, imag=0.000)
#         p.calculator.create_material('Material2')
#         p.calculator.update_material('Material2', real=2.000, imag=0.000)
#         p.calculator.create_material('Material3')
#         p.calculator.update_material('Material3', real=4.000, imag=0.000)
#         p.calculator.create_model()
#         p.calculator.create_layer('Layer1')
#         p.calculator.assign_material_to_layer('Material1', 'Layer1')
#         p.calculator.create_layer('Layer2')
#         p.calculator.assign_material_to_layer('Material2', 'Layer2')
#         p.calculator.update_layer('Layer2', thickness=10, sigma=1.0)
#         p.calculator.create_layer('Layer3')
#         p.calculator.assign_material_to_layer('Material3', 'Layer3')
#         p.calculator.update_layer('Layer3', sigma=1.0)
#         p.calculator.create_item('Item')
#         p.calculator.add_layer_to_item('Layer1', 'Item')
#         p.calculator.add_layer_to_item('Layer2', 'Item')
#         p.calculator.add_layer_to_item('Layer3', 'Item')
#         p.calculator.add_item('Item')
#         p.calculator.update_model('model', background=1e-7, resolution=5)
#         q = np.linspace(0.001, 0.3, 10)
#         expected = [
#             9.99956517e-01, 2.16286891e-03, 1.14086254e-04, 1.93031759e-05,
#             4.94188894e-06, 1.54191953e-06, 5.45592112e-07, 2.26619392e-07,
#             1.26726993e-07, 1.01842852e-07
#         ]
#         assert_allclose(p.fit_func(q), expected, rtol=0.04)

#     def test_calculate2(self):
#         p = BornAgain()
#         p.calculator.create_material('Material1')
#         p.calculator.update_material('Material1', real=0.000, imag=0.000)
#         p.calculator.create_material('Material2')
#         p.calculator.update_material('Material2', real=2.000, imag=0.000)
#         p.calculator.create_material('Material3')
#         p.calculator.update_material('Material3', real=4.000, imag=0.000)
#         p.calculator.create_model()
#         p.calculator.create_layer('Layer1')
#         p.calculator.assign_material_to_layer('Material1', 'Layer1')
#         p.calculator.create_layer('Layer2')
#         p.calculator.assign_material_to_layer('Material2', 'Layer2')
#         p.calculator.update_layer('Layer2', thickness=10, sigma=1.0)
#         p.calculator.create_layer('Layer3')
#         p.calculator.assign_material_to_layer('Material3', 'Layer3')
#         p.calculator.update_layer('Layer3', sigma=1.0)
#         p.calculator.create_item('Item1')
#         p.calculator.add_layer_to_item('Layer1', 'Item1')
#         p.calculator.create_item('Item2')
#         p.calculator.add_layer_to_item('Layer2', 'Item2')
#         p.calculator.add_layer_to_item('Layer1', 'Item2')
#         p.calculator.create_item('Item3')
#         p.calculator.add_layer_to_item('Layer3', 'Item3')
#         p.calculator.add_item('Item1')
#         p.calculator.add_item('Item2')
#         p.calculator.add_item('Item3')
#         p.calculator.update_item('Item2', repeats=10)
#         p.calculator.update_model('model', background=1e-7, resolution=5)
#         q = np.linspace(0.001, 0.3, 10)
#         expected = [1.000000e+00, 1.814452e-05, 1.225890e-04, 2.454331e-06,
#                     6.676318e-06, 8.362728e-07, 1.141096e-06, 4.090968e-07,
#                     3.489857e-07, 2.470789e-07]
#         assert_allclose(p.fit_func(q), expected, rtol=0.01)

#     def test_sld_profile(self):
#         p = BornAgain()
#         p.calculator.create_material('Material1')
#         p.calculator.update_material('Material1', real=0.000, imag=0.000)
#         p.calculator.create_material('Material2')
#         p.calculator.update_material('Material2', real=2.000, imag=0.000)
#         p.calculator.create_material('Material3')
#         p.calculator.update_material('Material3', real=4.000, imag=0.000)
#         p.calculator.create_model()
#         p.calculator.create_layer('Layer1')
#         p.calculator.assign_material_to_layer('Material1', 'Layer1')
#         p.calculator.create_layer('Layer2')
#         p.calculator.assign_material_to_layer('Material2', 'Layer2')
#         p.calculator.update_layer('Layer2', thickness=10, sigma=1.0)
#         p.calculator.create_layer('Layer3')
#         p.calculator.assign_material_to_layer('Material3', 'Layer3')
#         p.calculator.update_layer('Layer3', sigma=1.0)
#         p.calculator.create_item('Item')
#         p.calculator.add_layer_to_item('Layer1', 'Item')
#         p.calculator.add_layer_to_item('Layer2', 'Item')
#         p.calculator.add_layer_to_item('Layer3', 'Item')
#         p.calculator.add_item('Item')
#         assert_almost_equal(p.sld_profile()[1][0], 0)
#         assert_almost_equal(p.sld_profile()[1][-1], 4)
