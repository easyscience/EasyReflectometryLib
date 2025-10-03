__author__ = 'github.com/arm61'
__version__ = '0.0.1'

import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from orsopy.fileio import Header
from orsopy.fileio import load_orso

import easyreflectometry
from easyreflectometry.data.measurement import _load_orso
from easyreflectometry.data.measurement import _load_txt
from easyreflectometry.data.measurement import load
from easyreflectometry.data.measurement import load_as_dataset

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


class TestData(unittest.TestCase):
    def test_load_with_orso(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        er_data = load(fpath)
        o_data = load_orso(fpath)
        assert er_data['attrs']['R_spin_up']['orso_header'].value == Header.asdict(o_data[0].info)
        assert_almost_equal(er_data['data']['R_spin_up'].values, o_data[0].data[:, 1])
        assert_almost_equal(er_data['coords']['Qz_spin_up'].values, o_data[0].data[:, 0])
        assert_almost_equal(er_data['data']['R_spin_up'].variances, np.square(o_data[0].data[:, 2]))
        assert_almost_equal(er_data['coords']['Qz_spin_up'].variances, np.square(o_data[0].data[:, 3]))

    def test_load_with_txt(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        er_data = load(fpath)
        n_data = np.loadtxt(fpath)
        data_name = 'R_test_example1'
        coords_name = 'Qz_test_example1'
        assert_almost_equal(er_data['data'][data_name].values, n_data[:, 1])
        assert_almost_equal(er_data['coords'][coords_name].values, n_data[:, 0])
        assert_almost_equal(er_data['data'][data_name].variances, np.square(n_data[:, 2]))
        assert_almost_equal(er_data['coords'][coords_name].variances, np.square(n_data[:, 3]))

    def test_load_with_txt_commas(self):
        fpath = os.path.join(PATH_STATIC, 'ref_concat_1.txt')
        er_data = load(fpath)
        x, y, e = np.loadtxt(fpath, delimiter=',', comments='#', unpack=True)
        data_name = 'R_ref_concat_1'
        coords_name = 'Qz_ref_concat_1'
        assert_almost_equal(er_data['data'][data_name].values, y)
        assert_almost_equal(er_data['coords'][coords_name].values, x)
        assert_almost_equal(er_data['data'][data_name].variances, np.square(e))

    def test_orso1(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.ort')
        er_data = _load_orso(fpath)
        o_data = load_orso(fpath)
        assert er_data['attrs']['R_spin_up']['orso_header'].value == Header.asdict(o_data[0].info)
        assert_almost_equal(er_data['data']['R_spin_up'].values, o_data[0].data[:, 1])
        assert_almost_equal(er_data['coords']['Qz_spin_up'].values, o_data[0].data[:, 0])
        assert_almost_equal(er_data['data']['R_spin_up'].variances, np.square(o_data[0].data[:, 2]))
        assert_almost_equal(er_data['coords']['Qz_spin_up'].variances, np.square(o_data[0].data[:, 3]))

    def test_orso2(self):
        fpath = os.path.join(PATH_STATIC, 'test_example2.ort')
        er_data = _load_orso(fpath)
        o_data = load_orso(fpath)
        for i, o in enumerate(list(reversed(o_data))):
            assert er_data['attrs'][f'R_{o.info.data_set}']['orso_header'].value == Header.asdict(o.info)
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].values, o.data[:, 1])
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].values, o.data[:, 0])
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].variances, np.square(o.data[:, 2]))
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].variances, np.square(o.data[:, 3]))

    def test_orso3(self):
        fpath = os.path.join(PATH_STATIC, 'test_example3.ort')
        er_data = _load_orso(fpath)
        o_data = load_orso(fpath)
        for i, o in enumerate(o_data):
            assert er_data['attrs'][f'R_{o.info.data_set}']['orso_header'].value == Header.asdict(o.info)
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].values, o.data[:, 1])
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].values, o.data[:, 0])
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].variances, np.square(o.data[:, 2]))
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].variances, np.square(o.data[:, 3]))

    def test_orso4(self):
        fpath = os.path.join(PATH_STATIC, 'test_example4.ort')
        er_data = _load_orso(fpath)
        o_data = load_orso(fpath)
        for i, o in enumerate(o_data):
            print(list(er_data.keys()))
            assert er_data['attrs'][f'R_{o.info.data_set}']['orso_header'].value == Header.asdict(o.info)
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].values, o.data[:, 1])
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].values, o.data[:, 0])
            assert_almost_equal(er_data['data'][f'R_{o.info.data_set}'].variances, np.square(o.data[:, 2]))
            assert_almost_equal(er_data['coords'][f'Qz_{o.info.data_set}'].variances, np.square(o.data[:, 3]))

    def test_txt(self):
        fpath = os.path.join(PATH_STATIC, 'test_example1.txt')
        er_data = _load_txt(fpath)
        n_data = np.loadtxt(fpath)
        data_name = 'R_test_example1'
        coords_name = 'Qz_test_example1'
        assert_almost_equal(er_data['data'][data_name].values, n_data[:, 1])
        assert_almost_equal(er_data['coords'][coords_name].values, n_data[:, 0])
        assert_almost_equal(er_data['data'][data_name].variances, np.square(n_data[:, 2]))
        assert_almost_equal(er_data['coords'][coords_name].variances, np.square(n_data[:, 3]))
