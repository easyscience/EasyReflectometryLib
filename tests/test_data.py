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

    def test_zero_variance_filtering(self):
        """Test that data points with zero variance are properly filtered out."""
        import warnings
        
        fpath = os.path.join(PATH_STATIC, 'ref_zero_var.txt')
        
        # Load original data to verify expectations
        original_data = np.loadtxt(fpath, delimiter=',', comments='#')
        original_count = len(original_data)
        zero_variance_count = np.sum(original_data[:, 2] == 0.0)
        expected_filtered_count = original_count - zero_variance_count
        
        # Test with warnings captured
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset = load_as_dataset(fpath)
            
            # Check that a warning about removed points was issued
            warning_messages = [str(warning.message) for warning in w]
            zero_variance_warning = None
            for msg in warning_messages:
                if f"Removed {zero_variance_count} data point(s)" in msg:
                    zero_variance_warning = msg
                    break
            self.assertIsNotNone(zero_variance_warning, f"Expected warning about removed points not found in: {warning_messages}")
            
        # Verify the correct number of points were filtered
        self.assertEqual(len(dataset.x), expected_filtered_count)
        self.assertEqual(len(dataset.y), expected_filtered_count)
        self.assertEqual(len(dataset.ye), expected_filtered_count)
        
        # Verify no zero variances remain in the filtered data
        self.assertTrue(np.all(dataset.ye > 0.0))
        
        # Verify the filtered data matches expected values (non-zero variance points)
        non_zero_mask = original_data[:, 2] != 0.0
        expected_x = original_data[non_zero_mask, 0]
        expected_y = original_data[non_zero_mask, 1]
        expected_ye = np.square(original_data[non_zero_mask, 2])
        
        assert_almost_equal(dataset.x, expected_x)
        assert_almost_equal(dataset.y, expected_y)
        assert_almost_equal(dataset.ye, expected_ye)
