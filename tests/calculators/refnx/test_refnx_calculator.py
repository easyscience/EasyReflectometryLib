# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Refnx calculator, which implements the stateless contract."""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from easyreflectometry.calculators.refnx.calculator import Refnx
from easyreflectometry.calculators.refnx.stateless_wrapper import RefnxStatelessWrapper
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample


@pytest.fixture
def clear():
    global_object.map._clear()
    yield
    global_object.map._clear()


def three_layer_model(resolution: float = 5.0) -> Model:
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
    return Model(sample, 1.0, 0.0, PercentageFwhm(resolution), 'MyModel')


class TestContract:
    def test_name(self, clear):
        assert Refnx().name == 'refnx'

    def test_declares_the_contract(self, clear):
        # `root_type` is what the easyscience interface factory checks. It must
        # be on the calculator class, not on the wrapper it delegates to.
        assert Refnx.root_type is Model
        assert isinstance(Refnx()._wrapper, RefnxStatelessWrapper)

    def test_has_no_mirrored_state(self, clear):
        # Given
        calculator = Refnx()

        # Then no storage and no name-link dictionaries survive
        assert not hasattr(calculator._wrapper, 'storage')
        for attribute in ('_material_link', '_layer_link', '_item_link', '_model_link'):
            assert not hasattr(calculator, attribute)

    def test_reconcile_is_empty_for_a_write_through_backend(self, clear):
        assert Refnx().reconcile() == {}

    def test_the_mirroring_api_is_a_no_op(self, clear):
        # Given a model whose reflectivity is known
        calculator = Refnx()
        model = three_layer_model()
        calculator.set_model(model)
        q = np.linspace(0.001, 0.3, 10)
        before = calculator.reflectivity_profile(q, model.unique_name)

        # When the model code calls the mirroring API, as it does on every edit
        calculator.reset_storage()
        calculator.assign_material_to_layer('material', 'layer')
        calculator.add_layer_to_item('layer', 'item')
        calculator.remove_layer_from_item('layer', 'item')
        calculator.add_item_to_model('item', model.unique_name)
        calculator.remove_item_from_model('item', model.unique_name)

        # Then nothing changed, the calculator has no state to mirror into
        assert_array_equal(calculator.reflectivity_profile(q, model.unique_name), before)

    def test_deprecated_alias(self, clear):
        # Given
        calculator = Refnx()
        model = three_layer_model()
        calculator.set_model(model)
        q = np.linspace(0.001, 0.3, 10)

        # When
        with pytest.deprecated_call():
            aliased = calculator.reflectity_profile(q, model.unique_name)

        # Then
        assert_array_equal(aliased, calculator.reflectivity_profile(q, model.unique_name))


class TestModelRegistry:
    def test_set_model_registers_by_unique_name(self, clear):
        # Given
        calculator = Refnx()
        model = three_layer_model()

        # When
        calculator.set_model(model)

        # Then
        assert calculator.models == {model.unique_name: model}

    def test_several_models_are_kept_side_by_side(self, clear):
        # Given
        calculator = Refnx()
        first = three_layer_model()
        second = three_layer_model()
        second.sample[0].layers[1].thickness.value = 100.0

        # When
        calculator.set_model(first)
        calculator.set_model(second)

        # Then each model is evaluated from its own structure
        q = np.linspace(0.001, 0.3, 10)
        assert not np.array_equal(
            calculator.reflectivity_profile(q, first.unique_name),
            calculator.reflectivity_profile(q, second.unique_name),
        )


class TestEvaluation:
    def test_reflectivity_profile(self, clear):
        # Given
        calculator = Refnx()
        calculator.set_model(three_layer_model())
        q = np.linspace(0.001, 0.3, 10)

        # When/Then, the values the legacy storage-based calculator produced
        expected = [
            9.99956517e-01,
            2.16286891e-03,
            1.14086254e-04,
            1.93031759e-05,
            4.94188894e-06,
            1.54191953e-06,
            5.45592112e-07,
            2.26619392e-07,
            1.26726993e-07,
            1.01842852e-07,
        ]
        assert_almost_equal(calculator.reflectivity_profile(q), expected)

    def test_reflectivity_profile_with_repeats(self, clear):
        # Given the three-item model the legacy tests pinned, the middle
        # assembly repeating ten times
        ambient = Material(0.0, 0.0, 'Material1')
        film = Material(2.0, 0.0, 'Material2')
        substrate = Material(4.0, 0.0, 'Material3')
        ambient_layer = Layer(ambient, 0.0, 0.0, 'Layer1')
        film_layer = Layer(film, 10.0, 1.0, 'Layer2')
        substrate_layer = Layer(substrate, 0.0, 1.0, 'Layer3')
        sample = Sample(
            Multilayer(ambient_layer, 'Item1'),
            RepeatingMultilayer([film_layer, ambient_layer], 10, 'Item2'),
            Multilayer(substrate_layer, 'Item3'),
        )
        model = Model(sample, 1.0, 0.0, PercentageFwhm(5.0), 'MyModel')
        calculator = Refnx()
        calculator.set_model(model)
        q = np.linspace(0.001, 0.3, 10)

        # When/Then
        expected = [
            9.9995652e-01,
            1.7096697e-05,
            1.2253047e-04,
            2.4026928e-06,
            6.7117546e-06,
            8.3209877e-07,
            1.1512901e-06,
            4.1468151e-07,
            3.4981523e-07,
            2.5424356e-07,
        ]
        assert_almost_equal(calculator.reflectivity_profile(q, model.unique_name), expected)

    def test_sld_profile(self, clear):
        # Given
        calculator = Refnx()
        model = three_layer_model()
        calculator.set_model(model)

        # When
        sld = calculator.sld_profile(model.unique_name)

        # Then
        assert_almost_equal(sld[1][0], 0)
        assert_almost_equal(sld[1][-1], 4)

    def test_the_model_is_read_at_evaluation_time(self, clear):
        # Given
        calculator = Refnx()
        model = three_layer_model()
        calculator.set_model(model)
        q = np.linspace(0.001, 0.3, 10)
        before = calculator.reflectivity_profile(q, model.unique_name)

        # When a parameter changes, with no call into the calculator
        model.sample[0].layers[1].thickness.value = 50.0

        # Then
        assert not np.array_equal(calculator.reflectivity_profile(q, model.unique_name), before)


class TestConfiguration:
    def test_set_resolution_function(self, clear):
        # Given
        calculator = Refnx()
        resolution_function = PercentageFwhm(2.0)

        # When
        calculator.set_resolution_function(resolution_function)

        # Then
        assert calculator._wrapper._resolution_function is resolution_function

    def test_magnetism_flag(self, clear):
        # Given
        calculator = Refnx()

        # When
        calculator.include_magnetism = True

        # Then
        assert calculator.include_magnetism is True
