# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Refl1d calculator."""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from easyreflectometry.calculators.refl1d.calculator import Refl1d
from easyreflectometry.calculators.refl1d.stateless_wrapper import Refl1dStatelessWrapper
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.001, 0.3, 10)


@pytest.fixture
def clear():
    global_object.map._clear()
    yield
    global_object.map._clear()


def three_layer_model() -> Model:
    """Ambient | 10 A film | substrate, the model the legacy tests pinned."""
    ambient = Material(0.0, 0.0, 'Material1')
    film = Material(2.0, 0.0, 'Material2')
    substrate = Material(4.0, 0.0, 'Material3')
    sample = Sample(
        Multilayer(
            [
                Layer(ambient, 0.0, 0.0, 'Layer1'),
                Layer(film, 10.0, 1.0, 'Layer2'),
                Layer(substrate, 0.0, 1.0, 'Layer3'),
            ],
            'Item',
        )
    )
    return Model(sample, 1.0, 1e-7, PercentageFwhm(5.0), 'MyModel')


def three_item_model(repetitions: float = 10.0) -> Model:
    """The three-assembly model, the middle one repeating."""
    ambient = Material(0.0, 0.0, 'Material1')
    film = Material(2.0, 0.0, 'Material2')
    substrate = Material(4.0, 0.0, 'Material3')
    ambient_layer = Layer(ambient, 0.0, 0.0, 'Layer1')
    film_layer = Layer(film, 10.0, 1.0, 'Layer2')
    substrate_layer = Layer(substrate, 0.0, 1.0, 'Layer3')
    sample = Sample(
        Multilayer(ambient_layer, 'Item1'),
        RepeatingMultilayer([film_layer, ambient_layer], repetitions, 'Item2'),
        Multilayer(substrate_layer, 'Item3'),
    )
    return Model(sample, 1.0, 1e-7, PercentageFwhm(5.0), 'MyModel')


class TestContract:
    def test_name(self, clear):
        assert Refl1d().name == 'refl1d'

    def test_declares_the_contract(self, clear):
        assert Refl1d.root_type is Model
        assert isinstance(Refl1d()._wrapper, Refl1dStatelessWrapper)

    def test_has_no_mirrored_state(self, clear):
        # Given
        calculator = Refl1d()

        # Then no storage and no name-link dictionaries survive
        assert not hasattr(calculator._wrapper, 'storage')
        for attribute in ('_material_link', '_layer_link', '_item_link', '_model_link'):
            assert not hasattr(calculator, attribute)

    def test_reconcile_is_empty(self, clear):
        assert Refl1d().reconcile() == {}

    def test_the_mirroring_api_is_a_no_op(self, clear):
        # Given
        calculator = Refl1d()
        model = three_layer_model()
        calculator.set_model(model)
        before = calculator.reflectivity_profile(Q, model.unique_name)

        # When the model code calls the mirroring API, as it does on every edit
        calculator.reset_storage()
        calculator.assign_material_to_layer('material', 'layer')
        calculator.add_layer_to_item('layer', 'item')
        calculator.remove_layer_from_item('layer', 'item')
        calculator.add_item_to_model('item', model.unique_name)
        calculator.remove_item_from_model('item', model.unique_name)

        # Then
        assert_array_equal(calculator.reflectivity_profile(Q, model.unique_name), before)

    def test_deprecated_alias(self, clear):
        # Given
        calculator = Refl1d()
        model = three_layer_model()
        calculator.set_model(model)

        # When
        with pytest.deprecated_call():
            aliased = calculator.reflectity_profile(Q, model.unique_name)

        # Then
        assert_array_equal(aliased, calculator.reflectivity_profile(Q, model.unique_name))


class TestEvaluation:
    def test_reflectivity_profile(self, clear):
        # Given
        calculator = Refl1d()
        calculator.set_model(three_layer_model())

        # When/Then, the values the legacy storage-based calculator produced
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
            1.0520e-07,
        ]
        assert_almost_equal(calculator.reflectivity_profile(Q), expected, decimal=4)

    def test_reflectivity_profile_with_repeats(self, clear):
        # Given
        calculator = Refl1d()
        model = three_item_model()
        calculator.set_model(model)

        # When/Then
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
            2.7505e-07,
        ]
        assert_almost_equal(calculator.reflectivity_profile(Q, model.unique_name), expected, decimal=4)

    def test_reflectivity_profile_with_magnetism(self, clear):
        # Given
        calculator = Refl1d()
        calculator.include_magnetism = True
        model = three_item_model(repetitions=1.0)
        calculator.set_model(model)

        # When/Then
        expected = [
            9.99491251e-01,
            1.08413641e-02,
            1.46824402e-04,
            2.11783999e-05,
            5.24616472e-06,
            1.61422945e-06,
            5.66961121e-07,
            2.34269519e-07,
            1.30026616e-07,
            1.05139655e-07,
        ]
        assert_almost_equal(calculator.reflectivity_profile(Q, model.unique_name), expected, decimal=4)

    def test_sld_profile(self, clear):
        # Given
        calculator = Refl1d()
        model = three_layer_model()
        calculator.set_model(model)

        # When
        sld = calculator.sld_profile(model.unique_name)

        # Then
        assert_almost_equal(sld[1][0], 0)
        assert_almost_equal(sld[1][-1], 4)

    def test_the_model_is_read_at_evaluation_time(self, clear):
        # Given
        calculator = Refl1d()
        model = three_layer_model()
        calculator.set_model(model)
        before = calculator.reflectivity_profile(Q, model.unique_name)

        # When a parameter changes, with no call into the calculator
        model.sample[0].layers[1].thickness.value = 50.0

        # Then
        assert not np.array_equal(calculator.reflectivity_profile(Q, model.unique_name), before)


class TestConfiguration:
    def test_set_resolution_function(self, clear):
        # Given
        calculator = Refl1d()
        resolution_function = PercentageFwhm(2.0)

        # When
        calculator.set_resolution_function(resolution_function)

        # Then
        assert calculator._wrapper._resolution_function is resolution_function

    def test_magnetism_flag(self, clear):
        # Given
        calculator = Refl1d()

        # When
        calculator.include_magnetism = True

        # Then
        assert calculator.include_magnetism is True
