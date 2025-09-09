"""
Tests exercising the methods of the topmost classes for nested structure.
To ensure that the parameters are relayed.
"""

from copy import copy

from numpy.testing import assert_almost_equal

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import SurfactantLayer


def test_dict_skip_unique_name():
    # When
    resolution_function = LinearSpline([0, 10], [0, 10])
    model = Model(interface=CalculatorFactory())
    model.resolution_function = resolution_function
    model.add_assemblies(SurfactantLayer(), Multilayer(), RepeatingMultilayer())

    # Then
    dict_no_unique_name = model.as_dict(skip=['unique_name'])

    # Expect
    assert 'unique_name' not in dict_no_unique_name


def test_copy():
    # When
    resolution_function = LinearSpline([0, 10], [0, 10])
    model = Model(interface=CalculatorFactory())
    model.resolution_function = resolution_function
    model.add_assemblies(SurfactantLayer(), Multilayer(), RepeatingMultilayer())

    # Then
    model_copy = copy(model)

    # Expect
    assert sorted(model.as_dict()) == sorted(model_copy.as_dict())
    assert model._resolution_function.smearing(5.5) == model_copy._resolution_function.smearing(5.5)
    assert model.interface().name == model_copy.interface().name
    assert_almost_equal(
        model.interface().reflectity_profile([0.3], model.unique_name),
        model_copy.interface().reflectity_profile([0.3], model_copy.unique_name),
    )
    assert model.unique_name != model_copy.unique_name
    assert model.name == model_copy.name
    m1 = model.as_dict(skip=['interface', 'unique_name', 'resolution_function'])
    m2 = model_copy.as_dict(skip=['interface', 'unique_name', 'resolution_function'])
    assert m1 == m2
    # assert sorted(m1) == sorted(m2)
    # assert model.as_dict(skip=['interface', 'unique_name', 'resolution_function']) == model_copy.as_dict(
    #     skip=['interface', 'unique_name', 'resolution_function']
    # )
