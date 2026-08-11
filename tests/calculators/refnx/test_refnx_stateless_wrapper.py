# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the refnx wrapper, which builds its structure per evaluation.

Two things need pinning:

* **correctness** -- the rebuilt structure must reproduce the reference
  reflectivity of the `reflectivity/analysis validation suite
  <https://github.com/reflectivity/analysis>`_, and
* **write-through** -- every core ``Parameter`` handed to refnx must come back
  as a proxy which pushes backend-side assignments into easyscience.
"""

import gc
import weakref

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from refnx import reflect

from easyreflectometry.calculators.refnx.stateless_wrapper import RefnxStatelessWrapper
from easyreflectometry.calculators.refnx.stateless_wrapper import WriteBackParameter
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

from ._validation_data import test4_dat

Q = np.linspace(0.001, 0.3, 25)


@pytest.fixture
def clear():
    """Keep the global map free of leftovers from other test modules."""
    global_object.map._clear()
    yield
    global_object.map._clear()


def build_model(repetitions: float = 1.0, scale: float = 1.0, background: float = 0.0) -> Model:
    """Ambient | (film, ambient) x repetitions | substrate."""
    ambient = Material(0.0, 0.0, 'Ambient')
    film = Material(2.0, 0.0, 'Film')
    substrate = Material(4.0, 0.0, 'Substrate')
    ambient_layer = Layer(ambient, 0.0, 0.0, 'AmbientLayer')
    film_layer = Layer(film, 10.0, 1.0, 'FilmLayer')
    substrate_layer = Layer(substrate, 0.0, 1.0, 'SubstrateLayer')
    sample = Sample(
        Multilayer(ambient_layer, 'Fronting'),
        RepeatingMultilayer([film_layer, ambient_layer], repetitions, 'Stack'),
        Multilayer(substrate_layer, 'Backing'),
    )
    return Model(sample, scale, background, name='StatelessModel')


def stateless_wrapper(model: Model, resolution_function=None) -> RefnxStatelessWrapper:
    wrapper = RefnxStatelessWrapper()
    if resolution_function is not None:
        wrapper.set_resolution_function(resolution_function)
    wrapper.set_model(model)
    return wrapper


def validation_model(resolution_function=None) -> Model:
    """The four-layer model of the validation suite (test0 / test4)."""
    material1 = Material(2.070, 0.0, 'Material1')
    material2 = Material(3.450, 0.1, 'Material2')
    material3 = Material(5.000, 0.01, 'Material3')
    material4 = Material(6.000, 0.0, 'Material4')
    sample = Sample(
        Multilayer(Layer(material1, 0.0, 0.0, 'Layer1'), 'Item1'),
        Multilayer(Layer(material2, 100.0, 3.0, 'Layer2'), 'Item2'),
        Multilayer(Layer(material3, 200.0, 1.0, 'Layer3'), 'Item3'),
        Multilayer(Layer(material4, 0.0, 5.0, 'Layer4'), 'Item4'),
    )
    return Model(sample, 1.0, 0.0, resolution_function, 'ValidationModel')


class TestValidationSuite:
    """Reference values from https://github.com/reflectivity/analysis."""

    def test_unpolarised_test0(self, clear):
        # Given
        model = validation_model(PercentageFwhm(0))
        wrapper = stateless_wrapper(model, PercentageFwhm(0))
        q = np.array([
            5.000000000000000104e-03,
            3.717499999999999971e-02,
            5.449999999999999983e-02,
            1.005349999999999994e-01,
            2.955650000000000222e-01,
        ])

        # When/Then
        expected = [
            9.665000503913141472e-01,
            3.486325360684768590e-04,
            8.540420179439664689e-05,
            5.815959818366009312e-06,
            4.999742968030015832e-08,
        ]
        assert_almost_equal(wrapper.calculate(q, model.unique_name), expected)

    def test_unpolarised_test2(self, clear):
        # Given a bare interface
        material1 = Material(0.0, 0.0, 'Material1')
        material2 = Material(6.360, 0.0, 'Material2')
        sample = Sample(
            Multilayer(Layer(material1, 0.0, 0.0, 'Layer1'), 'Item1'),
            Multilayer(Layer(material2, 0.0, 3.0, 'Layer2'), 'Item2'),
        )
        model = Model(sample, 1.0, 0.0, PercentageFwhm(0), 'ValidationModel')
        wrapper = stateless_wrapper(model, PercentageFwhm(0))
        q = np.array([
            5.000000000000000104e-03,
            7.564500000000000390e-02,
            1.433050000000000157e-01,
            2.368350000000000177e-01,
            5.920499999999999652e-01,
        ])

        # When/Then
        expected = [
            1.000000000000000222e00,
            1.964576414578978456e-04,
            1.280698699505669096e-05,
            1.234290141526865827e-06,
            2.222536631965092181e-09,
        ]
        assert_almost_equal(wrapper.calculate(q, model.unique_name), expected)

    def test_unpolarised_test4_constant_resolution(self, clear):
        # Given
        model = validation_model(PercentageFwhm(5))
        wrapper = stateless_wrapper(model, PercentageFwhm(5))

        # When/Then
        assert_allclose(wrapper.calculate(test4_dat[:, 0], model.unique_name), test4_dat[:, 1], rtol=0.03)

    def test_unpolarised_test4_spline_resolution(self, clear):
        # Given, the file carries sigma, the spline is fed FWHM
        resolution_function = LinearSpline(test4_dat[:, 0], SIGMA_TO_FWHM * test4_dat[:, 3])
        model = validation_model(resolution_function)
        wrapper = stateless_wrapper(model, resolution_function)

        # When/Then
        assert_allclose(wrapper.calculate(test4_dat[:, 0], model.unique_name), test4_dat[:, 1], rtol=0.03)


class TestReferenceModels:
    def test_three_layer_reflectivity(self, clear):
        # Given, the model the legacy calculator tests pinned
        model = build_model()

        # When/Then
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
        assert_almost_equal(
            stateless_wrapper(model, PercentageFwhm(5.0)).calculate(np.linspace(0.001, 0.3, 10), model.unique_name),
            expected,
            decimal=6,
        )

    def test_sld_profile_ends(self, clear):
        # Given
        model = build_model()

        # When
        z, sld = stateless_wrapper(model).sld_profile(model.unique_name)

        # Then
        assert_almost_equal(sld[0], 0)
        assert_almost_equal(sld[-1], 4)

    def test_scale_and_background_are_taken_from_the_model(self, clear):
        # Given
        plain = build_model()
        scaled = build_model(scale=2.0, background=1e-3)

        # When
        plain_reflectivity = stateless_wrapper(plain).calculate(Q, plain.unique_name)
        scaled_reflectivity = stateless_wrapper(scaled).calculate(Q, scaled.unique_name)

        # Then
        assert_allclose(scaled_reflectivity, 2.0 * plain_reflectivity + 1e-3)

    def test_material_mixture_is_handled_like_a_material(self, clear):
        # Given a mixture, whose sld/isld are derived (dependent) parameters
        mixture = MaterialMixture(Material(2.0, 0.0), Material(6.0, 0.0), fraction=0.25)
        ambient = Material(0.0, 0.0, 'Ambient')
        sample = Sample(
            Multilayer(Layer(ambient, 0.0, 0.0, 'AmbientLayer'), 'Fronting'),
            Multilayer(Layer(mixture, 0.0, 1.0, 'MixtureLayer'), 'Backing'),
        )
        model = Model(sample, 1.0, 0.0)

        # When
        structure = RefnxStatelessWrapper()._build_structure(model)

        # Then the mixed sld is used, not either constituent's
        assert_almost_equal(structure.components[-1].sld.real.value, 3.0)


class TestStructureTraversal:
    def test_layer_and_assembly_order_is_preserved(self, clear):
        # Given
        model = build_model()

        # When
        structure = RefnxStatelessWrapper()._build_structure(model)

        # Then fronting -> substrate, in sample order
        assert [component.name for component in structure.components] == [
            model.sample[0].layers[0].unique_name,
            model.sample[1].layers[0].unique_name,
            model.sample[1].layers[1].unique_name,
            model.sample[2].layers[0].unique_name,
        ]

    def test_single_repetition_assemblies_are_flattened(self, clear):
        # Given
        model = build_model(repetitions=1.0)

        # When
        structure = RefnxStatelessWrapper()._build_structure(model)

        # Then
        assert all(isinstance(component, reflect.Slab) for component in structure.components)

    def test_repeated_assemblies_become_a_stack(self, clear):
        # Given
        model = build_model(repetitions=5.0)

        # When
        structure = RefnxStatelessWrapper()._build_structure(model)

        # Then
        stacks = [c for c in structure.components if isinstance(c, reflect.Stack)]
        assert len(stacks) == 1
        assert stacks[0].name == model.sample[1].unique_name
        assert stacks[0].repeats.value == 5.0
        assert len(stacks[0]) == 2

    def test_values_are_read_at_evaluation_time(self, clear):
        # Given
        model = build_model()
        wrapper = stateless_wrapper(model)
        before = wrapper.calculate(Q, model.unique_name)

        # When the model changes, with no call into the calculator at all
        model.sample[1].layers[0].thickness.value = 55.0

        # Then the next evaluation sees it
        after = wrapper.calculate(Q, model.unique_name)
        assert not np.array_equal(before, after)
        structure = wrapper._build_structure(model)
        assert structure.components[1].thick.value == 55.0

    def test_structural_edits_need_no_mirroring(self, clear):
        # Given
        model = build_model()
        wrapper = stateless_wrapper(model)
        assert len(wrapper._build_structure(model).components) == 4

        # When a layer is added through the model API only
        model.sample[1].add_layer(Layer(Material(3.0, 0.0, 'Extra'), 20.0, 1.0, 'ExtraLayer'))

        # Then
        assert len(wrapper._build_structure(model).components) == 5


class TestModelRegistry:
    def test_models_are_registered_by_unique_name(self, clear):
        # Given
        first = build_model()
        second = build_model()
        wrapper = RefnxStatelessWrapper()

        # When
        wrapper.set_model(first)
        wrapper.set_model(second)

        # Then
        assert wrapper.models == {
            first.unique_name: first,
            second.unique_name: second,
        }

    def test_the_sole_model_is_the_default(self, clear):
        # Given
        model = build_model()
        wrapper = stateless_wrapper(model)

        # When/Then
        assert_array_equal(wrapper.calculate(Q), wrapper.calculate(Q, model.unique_name))

    def test_several_models_require_a_model_id(self, clear):
        # Given
        wrapper = RefnxStatelessWrapper()
        wrapper.set_model(build_model())
        wrapper.set_model(build_model())

        # When/Then
        with pytest.raises(ValueError, match='pass `model_id`'):
            wrapper.calculate(Q)

    def test_each_model_is_evaluated_from_its_own_structure(self, clear):
        # Given
        thin = build_model()
        thick = build_model()
        thick.sample[1].layers[0].thickness.value = 100.0
        wrapper = RefnxStatelessWrapper()
        wrapper.set_model(thin)
        wrapper.set_model(thick)

        # When
        thin_reflectivity = wrapper.calculate(Q, thin.unique_name)
        thick_reflectivity = wrapper.calculate(Q, thick.unique_name)

        # Then
        assert not np.array_equal(thin_reflectivity, thick_reflectivity)
        assert_array_equal(thin_reflectivity, stateless_wrapper(thin).calculate(Q))

    def test_no_model_attached_raises(self, clear):
        # When/Then
        with pytest.raises(ValueError, match='No model is attached'):
            RefnxStatelessWrapper().calculate(Q)

    def test_unknown_model_id_raises(self, clear):
        # Given
        wrapper = stateless_wrapper(build_model())

        # When/Then
        with pytest.raises(ValueError, match='No model with unique name'):
            wrapper.calculate(Q, 'not_a_model')


class TestWriteBackProxies:
    def test_the_proxy_reads_the_current_core_value(self, clear):
        # Given
        model = build_model()
        thickness = model.sample[1].layers[0].thickness
        thickness.value = 42.0

        # When
        proxy = WriteBackParameter(thickness)

        # Then
        assert proxy.value == 42.0
        assert proxy.name == thickness.unique_name

    def test_a_backend_write_reaches_the_core_parameter(self, clear):
        # Given a free parameter, backends must not move fixed ones
        model = build_model()
        thickness = model.sample[1].layers[0].thickness
        thickness.fixed = False
        proxy = WriteBackParameter(thickness)

        # When refnx assigns, as a constraint or a refnx-side fitter would
        proxy.value = 33.0

        # Then
        assert thickness.value == 33.0

    def test_a_backend_write_notifies_observers(self, clear):
        # Given a dependent parameter downstream of the proxied one
        model = build_model()
        thickness = model.sample[1].layers[0].thickness
        thickness.fixed = False
        dependent = model.sample[2].layers[0].thickness
        dependent.make_dependent_on('2*a', {'a': thickness})
        proxy = WriteBackParameter(thickness)

        # When
        proxy.value = 20.0

        # Then the whole chain recalculates
        assert dependent.value == 40.0

    def test_a_backend_write_creates_no_undo_entry(self, clear):
        # Given
        model = build_model()
        thickness = model.sample[1].layers[0].thickness
        thickness.fixed = False
        proxy = WriteBackParameter(thickness)
        global_object.stack.clear()
        global_object.stack.enabled = True
        try:
            history_length = len(global_object.stack.history)

            # When
            proxy.value = 12.0

            # Then
            assert thickness.value == 12.0
            assert len(global_object.stack.history) == history_length
        finally:
            global_object.stack.enabled = False
            global_object.stack.clear()

    def test_a_refnx_constraint_writes_back_on_sync(self, clear):
        # Given two proxied parameters tied by a refnx-side constraint. refnx
        # evaluates a constraint lazily, on read, so it never passes through
        # the value setter and only the end-of-evaluation sync sees it.
        model = build_model()
        thickness = model.sample[1].layers[0].thickness
        roughness = model.sample[1].layers[0].roughness
        thickness.fixed = False
        roughness.fixed = False
        thickness_proxy = WriteBackParameter(thickness)
        roughness_proxy = WriteBackParameter(roughness)
        roughness_proxy.constraint = thickness_proxy / 10.0

        # When
        thickness_proxy.value = 50.0
        pushed = roughness_proxy.sync()

        # Then refnx recomputes the constrained parameter and core receives it
        assert pushed is True
        assert roughness_proxy.value == pytest.approx(5.0)
        assert roughness.value == pytest.approx(5.0)

    def test_sync_is_a_no_op_when_nothing_moved(self, clear):
        # Given, the normal case in a fit iteration
        model = build_model()
        thickness = model.sample[1].layers[0].thickness
        thickness.fixed = False
        proxy = WriteBackParameter(thickness)

        # When/Then
        assert proxy.sync() is False

    def test_a_constraint_reaches_core_through_a_full_evaluation(self, clear):
        # Given a constraint applied to the structure of one evaluation
        model = build_model()
        thickness = model.sample[1].layers[0].thickness
        roughness = model.sample[1].layers[0].roughness
        thickness.fixed = False
        roughness.fixed = False
        wrapper = stateless_wrapper(model)
        original_build = wrapper._build_structure

        def build_with_constraint(model, proxies=None):
            structure = original_build(model, proxies)
            slab = structure.components[1]
            slab.rough.constraint = slab.thick / 10.0
            return structure

        wrapper._build_structure = build_with_constraint

        # When
        wrapper.calculate(Q, model.unique_name)

        # Then the constrained value landed in the core parameter
        assert roughness.value == pytest.approx(thickness.value / 10.0)

    def test_a_dependent_core_parameter_rejects_the_write_back(self, clear, caplog):
        # Given a mixture, whose sld is dependent on the constituent fractions
        import logging

        mixture = MaterialMixture(Material(2.0, 0.0), Material(6.0, 0.0), fraction=0.25)
        proxy = WriteBackParameter(mixture._sld)

        # When
        with caplog.at_level(logging.WARNING, logger='easyscience.variable'):
            proxy.value = 99.0

        # Then core keeps the derived value and says why
        assert mixture.sld == pytest.approx(3.0)
        assert 'is dependent' in caplog.text

    @pytest.mark.parametrize(
        'attribute',
        ['thick', 'rough'],
    )
    def test_slab_constructor_keeps_the_proxy(self, clear, attribute):
        # Given
        model = build_model()
        wrapper = RefnxStatelessWrapper()

        # When
        slab = wrapper._slab(model.sample[1].layers[0])

        # Then
        assert isinstance(getattr(slab, attribute), WriteBackParameter)

    @pytest.mark.parametrize('attribute', ['real', 'imag'])
    def test_sld_assignment_keeps_the_proxy(self, clear, attribute):
        # Given
        model = build_model()

        # When
        sld = RefnxStatelessWrapper._sld(model.sample[1].layers[0].material)

        # Then
        assert isinstance(getattr(sld, attribute), WriteBackParameter)

    def test_stack_repeats_assignment_keeps_the_proxy(self, clear):
        # Given
        model = build_model(repetitions=5.0)

        # When
        structure = RefnxStatelessWrapper()._build_structure(model)

        # Then
        stack = [c for c in structure.components if isinstance(c, reflect.Stack)][0]
        assert isinstance(stack.repeats, WriteBackParameter)

    def test_reflect_model_constructor_keeps_the_proxies(self, clear):
        # Given
        model = build_model()
        captured = {}
        real_call = reflect.ReflectModel.__call__

        def spy(self, x, p=None, x_err=None):
            captured['scale'] = self.scale
            captured['bkg'] = self.bkg
            return real_call(self, x, p=p, x_err=x_err)

        # When
        reflect.ReflectModel.__call__ = spy
        try:
            stateless_wrapper(model).calculate(Q, model.unique_name)
        finally:
            reflect.ReflectModel.__call__ = real_call

        # Then
        assert isinstance(captured['scale'], WriteBackParameter)
        assert isinstance(captured['bkg'], WriteBackParameter)

    def test_a_float_assignment_replaces_the_proxy(self, clear):
        """Documents refnx's behaviour on the path we do *not* use.

        Assigning a float to ``slab.thick`` replaces the proxy with a plain
        refnx parameter, so that write would not reach easyscience. Every
        assignment ``_build_structure`` performs passes a proxy instance, which
        ``possibly_create_parameter`` lets through untouched; the blast radius
        of a replacement is one evaluation anyway, since the structure is
        rebuilt from scratch on the next call.
        """
        # Given
        model = build_model()
        slab = RefnxStatelessWrapper()._slab(model.sample[1].layers[0])

        # When
        slab.thick = 5.0

        # Then
        assert not isinstance(slab.thick, WriteBackParameter)
        assert model.sample[1].layers[0].thickness.value != 5.0

    def test_nothing_survives_the_evaluation(self, clear):
        # Given
        model = build_model()
        wrapper = stateless_wrapper(model)
        structure = wrapper._build_structure(model)
        proxies = [weakref.ref(structure.components[0].thick)]

        # When the structure is dropped
        del structure
        gc.collect()

        # Then no refnx-internal registry kept it alive
        assert all(proxy() is None for proxy in proxies)
        # ... and the wrapper still only holds the model
        assert list(vars(wrapper)) == ['_models', '_magnetism', '_resolution_function']


class TestResolution:
    def test_percentage_fwhm_is_passed_as_a_scalar(self, clear):
        # Given
        wrapper = stateless_wrapper(build_model(), PercentageFwhm(5.0))

        # When
        x_err = wrapper._resolution_vector(Q)

        # Then refnx reads a scalar x_err as a constant dq/q FWHM percentage
        assert np.ndim(x_err) == 0
        assert x_err == pytest.approx(5.0)

    def test_other_resolutions_are_converted_to_fwhm(self, clear):
        # Given
        q_knots = np.linspace(0.001, 0.5, 10)
        fwhm_knots = 0.02 * q_knots + 0.001
        wrapper = stateless_wrapper(build_model(), LinearSpline(q_knots, fwhm_knots))

        # When
        x_err = wrapper._resolution_vector(Q)

        # Then
        assert_allclose(x_err, np.interp(Q, q_knots, fwhm_knots))

    def test_smearing_is_sigma_and_is_scaled(self, clear):
        # Given
        q_knots = np.linspace(0.001, 0.5, 10)
        fwhm_knots = 0.02 * q_knots + 0.001
        resolution_function = LinearSpline(q_knots, fwhm_knots)
        wrapper = stateless_wrapper(build_model(), resolution_function)

        # When/Then
        assert_allclose(
            wrapper._resolution_vector(Q),
            resolution_function.smearing(Q) * SIGMA_TO_FWHM,
        )


class TestMagnetism:
    def test_magnetism_is_not_supported(self, clear):
        # When/Then
        with pytest.raises(NotImplementedError):
            RefnxStatelessWrapper().include_magnetism = True

    def test_the_flag_is_readable(self, clear):
        # When/Then
        assert RefnxStatelessWrapper().include_magnetism is False
