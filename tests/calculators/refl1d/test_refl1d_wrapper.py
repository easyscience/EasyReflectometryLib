"""
Tests for Refl1d wrapper.
"""

__author__ = 'github.com/arm61'
__version__ = '0.0.1'

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from easyreflectometry.calculators.refl1d.wrapper import Refl1dWrapper
from easyreflectometry.calculators.refl1d.wrapper import _build_sample
from easyreflectometry.calculators.refl1d.wrapper import _get_oversampling_q
from easyreflectometry.calculators.refl1d.wrapper import _get_polarized_probe
from easyreflectometry.calculators.refl1d.wrapper import _get_probe


class TestRefl1d(unittest.TestCase):
    def test_init(self):
        p = Refl1dWrapper()
        assert_equal(list(p.storage.keys()), ['material', 'layer', 'item', 'model'])
        assert_equal(issubclass(p.storage['material'].__class__, dict), True)
        assert p._magnetism is False

    def test_set_magnetism(self):
        p = Refl1dWrapper()
        p.magnetism = True
        assert p._magnetism is True

    def test_reset_storage(self):
        p = Refl1dWrapper()
        p.storage['material']['a'] = 1
        assert_equal(p.storage['material']['a'], 1)
        p.reset_storage()
        assert_equal(p.storage['material'], {})

    def test_create_material(self):
        p = Refl1dWrapper()
        p.create_material('Si')
        assert_equal(list(p.storage['material'].keys()), ['Si'])
        assert_almost_equal(p.storage['material']['Si'].rho.value, 0.0)
        assert_almost_equal(p.storage['material']['Si'].irho.value, 0.0)
        assert_equal(p.storage['material']['Si'].name, 'Si')

    def test_update_material(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        assert_equal(list(p.storage['material'].keys()), ['B'])
        assert_almost_equal(p.storage['material']['B'].rho.value, 6.908)
        assert_almost_equal(p.storage['material']['B'].irho.value, -0.278)

    def test_get_material_value(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        assert_equal(list(p.storage['material'].keys()), ['B'])
        assert_almost_equal(p.get_material_value('B', 'rho'), 6.908)
        assert_almost_equal(p.get_material_value('B', 'irho'), -0.278)

    def test_create_layer(self):
        p = Refl1dWrapper()
        p.create_layer('Si')
        assert_equal(list(p.storage['layer'].keys()), ['Si'])
        assert_almost_equal(p.storage['layer']['Si'].thickness.value, 0)
        assert_almost_equal(p.storage['layer']['Si'].interface.value, 0)

    def test_update_layer(self):
        p = Refl1dWrapper()
        p.create_layer('Si')
        p.update_layer('Si', thickness=10, interface=5)
        assert_almost_equal(p.storage['layer']['Si'].thickness.value, 10)
        assert_almost_equal(p.storage['layer']['Si'].interface.value, 5)

    def test_update_magnetic_layer(self):
        p = Refl1dWrapper()
        p.magnetism = True
        p.create_layer('Si')
        p.update_layer('Si', magnetism_rhoM=5, magnetism_thetaM=10)
        assert_almost_equal(p.storage['layer']['Si'].magnetism.thetaM.value, 10)
        assert_almost_equal(p.storage['layer']['Si'].magnetism.rhoM.value, 5)

    def test_get_layer_value(self):
        p = Refl1dWrapper()
        p.create_layer('Si')
        p.update_layer('Si', thickness=10, interface=5)
        assert_almost_equal(p.get_layer_value('Si', 'thickness'), 10)
        assert_almost_equal(p.get_layer_value('Si', 'interface'), 5)

    def test_magnetic_get_layer_value(self):
        p = Refl1dWrapper()
        p.magnetism = True
        p.create_layer('Si')
        p.update_layer('Si', magnetism_rhoM=5, magnetism_thetaM=10)
        assert_almost_equal(p.get_layer_value('Si', 'magnetism_thetaM'), 10)
        assert_almost_equal(p.get_layer_value('Si', 'magnetism_rhoM'), 5)

    def test_create_item(self):
        p = Refl1dWrapper()
        p.create_item('SiNi')
        assert_equal(list(p.storage['item'].keys()), ['SiNi'])
        assert_almost_equal(p.storage['item']['SiNi'].repeat.value, 1)

    def test_update_item(self):
        p = Refl1dWrapper()
        p.create_item('SiNi')
        p.update_item('SiNi', repeat=10)
        assert_almost_equal(p.storage['item']['SiNi'].repeat.value, 10)

    def test_get_item_value(self):
        p = Refl1dWrapper()
        p.create_item('SiNi')
        p.update_item('SiNi', repeat=10)
        assert_almost_equal(p.get_item_value('SiNi', 'repeat'), 10)

    def test_create_model(self):
        p = Refl1dWrapper()
        p.create_model('MyModel')
        assert_equal(p.storage['model']['MyModel'], {'scale': 1, 'bkg': 0, 'items': []})

    def test_update_model(self):
        p = Refl1dWrapper()
        p.create_model('MyModel')
        p.update_model('MyModel', scale=2, bkg=1e-3)
        assert_almost_equal(p.storage['model']['MyModel']['scale'], 2)
        assert_almost_equal(p.storage['model']['MyModel']['bkg'], 1e-3)

    def test_get_model_value(self):
        p = Refl1dWrapper()
        p.create_model('MyModel')
        p.update_model('MyModel', scale=2, bkg=1e-3)
        assert_almost_equal(p.get_model_value('MyModel', 'scale'), 2)
        assert_almost_equal(p.get_model_value('MyModel', 'bkg'), 1e-3)

    def test_assign_material_to_layer(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        assert_almost_equal(p.storage['layer']['B_layer'].material.rho.value, 6.908)
        assert_almost_equal(p.storage['layer']['B_layer'].material.irho.value, -0.278)

    def test_add_layer_to_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 0)
        p.add_layer_to_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 1)
        assert_equal(p.storage['item']['B_item'][0].name, 'B_layer')

    def test_add_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        p.create_model('MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 0)
        p.add_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 1)
        assert_equal(p.storage['model']['MyModel']['items'][0].name, 'B_item')

    def test_remove_layer_from_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 1)
        p.remove_layer_from_item('B_layer', 'B_item')
        assert_equal(len(p.storage['item']['B_item'].stack), 0)

    def test_remove_item(self):
        p = Refl1dWrapper()
        p.create_material('B')
        p.update_material('B', rho=6.908, irho=-0.278)
        p.create_layer('B_layer')
        p.assign_material_to_layer('B', 'B_layer')
        p.create_item('B_item')
        p.add_layer_to_item('B_layer', 'B_item')
        p.create_model('MyModel')
        p.add_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 1)
        p.remove_item('B_item', 'MyModel')
        assert_equal(len(p.storage['model']['MyModel']['items']), 0)

    def test_calculate(self):
        p = Refl1dWrapper()
        p.create_material('Material1')
        p.update_material('Material1', rho=0.000, irho=0.000)
        p.create_material('Material2')
        p.update_material('Material2', rho=2.000, irho=0.000)
        p.create_material('Material3')
        p.update_material('Material3', rho=4.000, irho=0.000)
        p.create_model('MyModel')
        p.update_model('MyModel', bkg=1e-7)
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thickness=10, interface=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', interface=1.0)
        p.create_item('Item')
        p.add_layer_to_item('Layer1', 'Item')
        p.add_layer_to_item('Layer2', 'Item')
        p.add_layer_to_item('Layer3', 'Item')
        p.add_item('Item', 'MyModel')
        q = np.linspace(0.001, 0.3, 10)
        expected = [
            9.9949e-01,
            1.0842e-02,
            1.4709e-04,
            2.1277e-05,
            5.2902e-06,
            1.6347e-06,
            5.7605e-07,
            2.3775e-07,
            1.3093e-07,
            1.0520e-07
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected, decimal=4)

    def test_calculate_three_items(self):
        p = Refl1dWrapper()
        p.create_material('Material1')
        p.update_material('Material1', rho=0.000, irho=0.000)
        p.create_material('Material2')
        p.update_material('Material2', rho=2.000, irho=0.000)
        p.create_material('Material3')
        p.update_material('Material3', rho=4.000, irho=0.000)
        p.create_model('MyModel')
        p.update_model('MyModel', bkg=1e-7)
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thickness=10, interface=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', interface=1.0)
        p.create_item('Item1')
        p.add_layer_to_item('Layer1', 'Item1')
        p.create_item('Item2')
        p.add_layer_to_item('Layer2', 'Item2')
        p.add_layer_to_item('Layer1', 'Item2')
        p.create_item('Item3')
        p.add_layer_to_item('Layer3', 'Item3')
        p.add_item('Item1', 'MyModel')
        p.add_item('Item2', 'MyModel')
        p.add_item('Item3', 'MyModel')
        p.update_item('Item2', repeat=10)
        q = np.linspace(0.001, 0.3, 10)
        expected = [
            9.9949e-01,
            8.7414e-03,
            1.1850e-04,
            5.4758e-06,
            6.3826e-06,
            1.0777e-06,
            1.0968e-06,
            4.5635e-07,
            3.4120e-07,
            2.7505e-07
        ]
        assert_almost_equal(p.calculate(q, 'MyModel'), expected, decimal=4)

    def test_sld_profile(self):
        p = Refl1dWrapper()
        p.create_material('Material1')
        p.update_material('Material1', rho=0.000, irho=0.000)
        p.create_material('Material2')
        p.update_material('Material2', rho=2.000, irho=0.000)
        p.create_material('Material3')
        p.update_material('Material3', rho=4.000, irho=0.000)
        p.create_model('MyModel')
        p.create_layer('Layer1')
        p.assign_material_to_layer('Material1', 'Layer1')
        p.create_layer('Layer2')
        p.assign_material_to_layer('Material2', 'Layer2')
        p.update_layer('Layer2', thickness=10, interface=1.0)
        p.create_layer('Layer3')
        p.assign_material_to_layer('Material3', 'Layer3')
        p.update_layer('Layer3', interface=1.0)
        p.create_item('Item')
        p.add_layer_to_item('Layer1', 'Item')
        p.add_layer_to_item('Layer2', 'Item')
        p.add_layer_to_item('Layer3', 'Item')
        p.add_item('Item', 'MyModel')
        assert_almost_equal(p.sld_profile('MyModel')[1][0], 0)
        assert_almost_equal(p.sld_profile('MyModel')[1][-1], 4)


def test_get_oversampling():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)

    # Then
    oversampling = _get_oversampling_q(q_array=q, dq_array=dq, oversampling_factor=5)

    # Expect
    assert len(oversampling) == 50
    assert oversampling[0] == 0.965
    assert oversampling[-1] == 10.35


def test_get_probe():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage)

    # Then
    assert all(probe.Q == q)
    assert all(probe.calc_Q == q)
    assert all(probe.dQ == dq)
    assert probe.intensity.value == 10
    assert probe.background.value == 20


def test_get_probe_oversampling():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage, oversampling_factor=2)

    # Then
    assert len(probe.calc_Q) == len(q)


def test_get_polarized_probe():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_polarized_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage)

    # Then
    assert all(probe.Q == q)
    assert all(probe.calc_Q == q)
    assert all(probe.dQ == dq)
    assert len(probe.calc_Q) == len(q)
    assert len(probe.xs) == 4
    assert probe.xs[1:4] == [None, None, None]
    assert probe.xs[0].intensity.value == 10
    assert probe.xs[0].background.value == 20


def test_get_polarized_probe_oversampling():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_polarized_probe(q_array=q, dq_array=dq, model_name=model_name, storage=storage, oversampling_factor=2)

    # Then
    assert len(probe.xs[0].calc_Q) == 2 * len(q)


def test_get_polarized_probe_polarization():
    # When
    q = np.linspace(1, 10, 10)
    dq = np.linspace(0.01, 0.1, 10)
    model_name = 'model_name'

    storage = {'model': {model_name: {}}}
    storage['model'][model_name]['scale'] = 10.0
    storage['model'][model_name]['bkg'] = 20.0

    # Then
    probe = _get_polarized_probe(
        q_array=q,
        dq_array=dq,
        model_name=model_name,
        storage=storage,
        all_polarizations=True,
    )

    # Expect
    assert len(probe.xs[0].calc_Q) == len(q)
    assert len(probe.xs[1].calc_Q) == len(q)
    assert len(probe.xs[2].calc_Q) == len(q)
    assert len(probe.xs[3].calc_Q) == len(q)


@patch('easyreflectometry.calculators.refl1d.wrapper.names.Stack')
@patch('easyreflectometry.calculators.refl1d.wrapper.Repeat')
def test_build_sample(mock_repeat, mock_stack):
    # When
    mock_item_1 = MagicMock()
    mock_item_1.repeat = MagicMock()
    mock_item_1.repeat.value = 1
    mock_item_1.stack = ['1a', '1b']
    mock_item_2 = MagicMock()
    mock_item_2.repeat = MagicMock()
    mock_item_2.repeat.value = 2
    mock_item_2.stack = ['2a', '2b']
    model_name = 'model_name'
    mock_stack.__or__ = MagicMock()

    storage = {'model': {model_name: {'items': []}}}
    storage['model'][model_name]['items'].append(mock_item_1)
    storage['model'][model_name]['items'].append(mock_item_2)

    # Then
    _ = _build_sample(model_name=model_name, storage=storage)

    # Expect
    assert mock_stack.call_count == 2
    assert mock_repeat.call_count == 1
    # TODO do asserts on sample returned by _build_sample
    # will probably use other build_sample function in future
    # difficult to test current implementation
