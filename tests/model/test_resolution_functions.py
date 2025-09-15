import unittest

import numpy as np

from easyreflectometry.model.resolution_functions import DEFAULT_RESOLUTION_FWHM_PERCENTAGE
from easyreflectometry.model.resolution_functions import LinearSpline
from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.model.resolution_functions import Pointwise
from easyreflectometry.model.resolution_functions import ResolutionFunction


class TestPercentageFwhm(unittest.TestCase):
    def test_constructor(self):
        # When
        resolution_function = PercentageFwhm(1.0)

        # Then Expect
        assert np.all(resolution_function.smearing([0, 2.5]) == np.array([1.0, 1.0]))
        assert resolution_function.smearing([-100]) == np.array([1.0])
        assert resolution_function.smearing([100]) == np.array([1.0])

    def test_constructor_none(self):
        # When
        resolution_function = PercentageFwhm()

        # Then Expect
        assert np.all(
            resolution_function.smearing([0, 2.5]) == [DEFAULT_RESOLUTION_FWHM_PERCENTAGE, DEFAULT_RESOLUTION_FWHM_PERCENTAGE]
        )
        assert resolution_function.smearing([-100]) == DEFAULT_RESOLUTION_FWHM_PERCENTAGE
        assert resolution_function.smearing([100]) == DEFAULT_RESOLUTION_FWHM_PERCENTAGE

    def test_as_dict(self):
        # When
        resolution_function = PercentageFwhm(1.0)

        # Then Expect
        resolution_function.as_dict() == {'smearing': 'PercentageFwhm', 'constant': 1.0}

    def test_dict_round_trip(self):
        # When
        expected_resolution_function = PercentageFwhm(1.0)
        res_dict = expected_resolution_function.as_dict()

        # Then
        resolution_function = ResolutionFunction.from_dict(res_dict)

        # Expect
        assert all(resolution_function.smearing([0, 2.5]) == expected_resolution_function.smearing([0, 2.5]))


class TestLinearSpline(unittest.TestCase):
    def test_constructor(self):
        # When
        resolution_function = LinearSpline(q_data_points=[0, 10], fwhm_values=[5, 10])

        # Then Expect
        assert np.all(resolution_function.smearing([0, 2.5]) == np.array([5, 6.25]))
        assert resolution_function.smearing([-100]) == np.array([5.0])
        assert resolution_function.smearing([100]) == np.array([10.0])

    def test_as_dict(self):
        # When
        resolution_function = LinearSpline(q_data_points=[0, 10], fwhm_values=[5, 10])

        # Then Expect
        resolution_function.as_dict() == {'smearing': 'LinearSpline', 'q_data_points': [0, 10], 'fwhm_values': [5, 10]}

    def test_dict_round_trip(self):
        # When
        expected_resolution_function = LinearSpline(q_data_points=[0, 10], fwhm_values=[5, 10])
        res_dict = expected_resolution_function.as_dict()

        # Then
        resolution_function = ResolutionFunction.from_dict(res_dict)

        # Expect
        assert all(resolution_function.smearing([0, 2.5]) == expected_resolution_function.smearing([0, 2.5]))

class TestPointwise(unittest.TestCase):

    data_points = []
    data_points.append([0.1, 0.2, 0.3, 0.4, 0.5]) # Qz
    data_points.append([1.1, 2.2, 3.3, 4.4, 5.5]) # R
    data_points.append([0.03, 0.04, 0.05, 0.06, 0.07]) # sQz
    def test_constructor(self):

        # When
        resolution_function = Pointwise(q_data_points=self.data_points)

        # Then Expect
        assert np.allclose(np.array(resolution_function.smearing()),
                           np.array([2.51664683, 2.84038734, 3.2460762 , 3.6796519 , 4.07869271]))

    def test_as_dict(self):
        # When
        resolution_function = Pointwise(q_data_points=self.data_points)

        # Then Expect
        assert resolution_function.as_dict(), {'smearing': 'Pointwise', 'q_data_points': [0, 10]}

    def test_dict_round_trip(self):
        # When
        expected_resolution_function = Pointwise(q_data_points=self.data_points)
        res_dict = expected_resolution_function.as_dict()

        # Then
        resolution_function = ResolutionFunction.from_dict(res_dict)

        # Expect
        assert all(resolution_function.smearing() == expected_resolution_function.smearing())
