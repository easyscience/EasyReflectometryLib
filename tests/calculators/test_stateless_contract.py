# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the calculators behind the factory.

Both calculators read the model at evaluation time and keep no copy of it, so
switching between them, sharing one factory across models, and serializing a
model with either attached all have to work.
"""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

from easyreflectometry.calculators import CalculatorBase
from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators.refl1d.calculator import Refl1d
from easyreflectometry.calculators.refnx.calculator import Refnx
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.005, 0.3, 20)


@pytest.fixture
def clear():
    global_object.map._clear()
    yield
    global_object.map._clear()


def assert_unbound(parameter) -> None:
    """Assert no backend callback is wired to the parameter.

    Current easyscience always gives a Parameter an inert `_callback`
    (`property()` with no fget/fset); a bound parameter is one whose callback
    has accessors wired to a backend object. Future cores drop the attribute
    altogether, which this accepts too.
    """
    callback = getattr(parameter, '_callback', None)
    assert callback is None or (callback.fget is None and callback.fset is None)


def switch_calculator(model: Model, name: str) -> None:
    """Switch the calculator and re-bind, the way `Project.calculator` does.

    `BaseCore` has no `switch_interface`, the library switches the shared
    factory and then re-binds every model.
    """
    model.interface.switch(name)
    model.interface.reset_storage()
    model.generate_bindings()


def build_model(thickness: float = 100.0, name: str = 'Model') -> Model:
    ambient = Material(0.0, 0.0, 'Ambient')
    film = Material(3.45, 0.0, 'Film')
    substrate = Material(2.07, 0.0, 'Substrate')
    sample = Sample(
        Multilayer(
            [
                Layer(ambient, 0.0, 0.0, 'AmbientLayer'),
                Layer(film, thickness, 3.0, 'FilmLayer'),
                Layer(substrate, 0.0, 3.0, 'SubstrateLayer'),
            ],
            'Item',
        )
    )
    return Model(sample, 1.0, 0.0, PercentageFwhm(1.0), name)


class TestRegistration:
    def test_both_generations_are_discovered(self, clear):
        # Given
        factory = CalculatorFactory()

        # Then
        assert 'refnx' in factory.available_interfaces
        assert 'refl1d' in factory.available_interfaces

    def test_every_calculator_registers_itself(self, clear):
        # Then
        for calculator in (Refnx, Refl1d):
            assert issubclass(calculator, CalculatorBase)
            assert calculator in CalculatorBase._calculators

    def test_the_base_is_not_registered(self, clear):
        # Then
        assert CalculatorBase not in CalculatorBase._calculators


class TestBinding:
    def test_only_the_model_root_is_registered(self, clear):
        # Given
        model = build_model()
        factory = CalculatorFactory()

        # When
        model.interface = factory

        # Then the interface reaches every child, but only the root is kept
        assert list(factory().models) == [model.unique_name]

    def test_no_parameter_is_bound_to_the_backend(self, clear):
        # Given
        model = build_model()

        # When
        model.interface = CalculatorFactory()

        # Then reading a value never reaches the calculator
        for parameter in model.get_all_parameters():
            assert_unbound(parameter)

    def test_one_factory_serves_several_models(self, clear):
        # Given, this is how `Project` attaches its calculator
        factory = CalculatorFactory()
        thin = build_model(thickness=50.0, name='Thin')
        thick = build_model(thickness=200.0, name='Thick')

        # When
        thin.interface = factory
        thick.interface = factory

        # Then each model keeps its own structure
        assert set(factory().models) == {thin.unique_name, thick.unique_name}
        assert not np.array_equal(
            factory().reflectivity_profile(Q, thin.unique_name),
            factory().reflectivity_profile(Q, thick.unique_name),
        )

    def test_the_fit_entry_point_routes_to_the_reflectivity(self, clear):
        # Given
        model = build_model()
        factory = CalculatorFactory()
        model.interface = factory

        # When, this is the call `MultiFitter` wraps per model
        through_fit_func = factory.fit_func(Q, model.unique_name)

        # Then
        assert_array_equal(through_fit_func, factory().reflectivity_profile(Q, model.unique_name))

    def test_structural_edits_need_no_interface_calls(self, clear):
        # Given
        model = build_model()
        factory = CalculatorFactory()
        model.interface = factory
        before = factory().reflectivity_profile(Q, model.unique_name)

        # When an assembly is added through the model API
        model.add_assemblies(Multilayer(Layer(Material(6.0, 0.0, 'Extra'), 40.0, 2.0, 'ExtraLayer')))

        # Then the next evaluation sees it
        assert not np.array_equal(factory().reflectivity_profile(Q, model.unique_name), before)


class TestSwitchingCalculators:
    def test_refnx_to_refl1d_preserves_values(self, clear):
        # Given
        model = build_model()
        model.interface = CalculatorFactory()
        refnx_reflectivity = model.interface().reflectivity_profile(Q, model.unique_name)

        # When
        switch_calculator(model, 'refl1d')

        # Then the values never left the parameters, so the other engine reads
        # exactly the same model and the two broadly agree
        assert model.sample[0].layers[1].thickness.value == 100.0
        refl1d_reflectivity = model.interface().reflectivity_profile(Q, model.unique_name)
        assert_almost_equal(np.log10(refl1d_reflectivity), np.log10(refnx_reflectivity), decimal=1)

    def test_refl1d_to_refnx_preserves_values(self, clear):
        # Given
        model = build_model()
        factory = CalculatorFactory()
        factory.switch('refl1d')
        model.interface = factory

        # When
        switch_calculator(model, 'refnx')

        # Then
        assert model.sample[0].layers[1].thickness.value == 100.0
        assert list(factory().models) == [model.unique_name]
        assert len(factory().reflectivity_profile(Q, model.unique_name)) == len(Q)

    def test_nothing_is_bound_to_the_discarded_backend(self, clear):
        # Given a model bound to one calculator
        model = build_model()
        factory = CalculatorFactory()
        factory.switch('refl1d')
        model.interface = factory

        # When switching to the other
        switch_calculator(model, 'refnx')

        # Then no parameter carries a link to the wrapper which was thrown away
        for parameter in model.get_all_parameters():
            assert_unbound(parameter)

    def test_easyscience_constraints_survive_a_round_trip(self, clear):
        # Given a dependency between two model parameters
        model = build_model()
        model.interface = CalculatorFactory()
        front = model.sample[0].layers[1].thickness
        back = model.sample[0].layers[2].roughness
        back.make_dependent_on('a/10', {'a': front})
        assert back.value == pytest.approx(10.0)

        # When switching both ways
        switch_calculator(model, 'refl1d')
        switch_calculator(model, 'refnx')

        # Then the dependency is intact and still live
        front.fixed = False
        front.value = 200.0
        assert back.value == pytest.approx(20.0)

    def test_a_bare_switch_leaves_an_empty_registry(self, clear):
        # Given
        model = build_model()
        factory = CalculatorFactory()
        model.interface = factory

        # When the factory is switched directly, with no follow-up re-binding
        factory.switch('refl1d')
        factory.switch('refnx')

        # Then evaluation refuses rather than using a stale model
        assert factory().models == {}
        with pytest.raises(ValueError, match='No model is attached'):
            factory().reflectivity_profile(Q)

        # ... and re-binding, which every real caller does, restores it
        model.generate_bindings()
        assert list(factory().models) == [model.unique_name]

    def test_multi_model_projects_switch_correctly(self, clear):
        # Given one factory shared by two models
        factory = CalculatorFactory()
        thin = build_model(thickness=50.0, name='Thin')
        thick = build_model(thickness=200.0, name='Thick')
        thin.interface = factory
        thick.interface = factory
        expected_thin = factory().reflectivity_profile(Q, thin.unique_name)

        # When the calculator is switched and the models re-bound, the way
        # `Project.calculator` does it
        factory.switch('refl1d')
        factory.reset_storage()
        for model in (thin, thick):
            model.generate_bindings()
        factory.switch('refnx')
        factory.reset_storage()
        for model in (thin, thick):
            model.generate_bindings()

        # Then each model still evaluates as itself
        assert_array_equal(factory().reflectivity_profile(Q, thin.unique_name), expected_thin)
        assert not np.array_equal(
            factory().reflectivity_profile(Q, thin.unique_name),
            factory().reflectivity_profile(Q, thick.unique_name),
        )


class TestSerialization:
    def test_round_trip_with_an_interface_attached(self, clear):
        # Given
        model = build_model()
        model.interface = CalculatorFactory()
        expected = model.interface().reflectivity_profile(Q, model.unique_name)
        model_dict = model.as_dict()

        # When
        global_object.map._clear()
        restored = Model.from_dict(model_dict)

        # Then
        assert restored.interface().name == 'refnx'
        assert list(restored.interface().models) == [restored.unique_name]
        assert_array_equal(restored.interface().reflectivity_profile(Q, restored.unique_name), expected)

    def test_the_calculator_is_restored_before_the_model_attaches(self, clear):
        # Given a freshly switched factory, which is the state `from_dict`
        # produces before it assigns `model.interface`
        factory = CalculatorFactory()
        factory.switch('refnx')

        # Then evaluation fails cleanly, there is no stale model to guess at
        with pytest.raises(ValueError, match='No model is attached'):
            factory().reflectivity_profile(Q)

        # When the model attaches afterwards
        model = build_model()
        model.interface = factory

        # Then
        assert len(factory().reflectivity_profile(Q, model.unique_name)) == len(Q)
