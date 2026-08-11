# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the refl1d wrapper, which builds its sample per evaluation."""

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from refl1d import names
from refl1d.sample.layers import Repeat

from easyreflectometry.calculators.refl1d.stateless_wrapper import Refl1dStatelessWrapper
from easyreflectometry.calculators.refl1d.stateless_wrapper import _get_oversampling_q
from easyreflectometry.calculators.refl1d.stateless_wrapper import _get_polarized_probe
from easyreflectometry.calculators.refl1d.stateless_wrapper import _get_probe
from easyreflectometry.model import LinearSpline
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import MaterialMixture
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.001, 0.3, 25)


@pytest.fixture
def clear():
    global_object.map._clear()
    yield
    global_object.map._clear()


def build_model(repetitions: float = 1.0, scale: float = 1.0, background: float = 1e-7) -> Model:
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
    return Model(sample, scale, background, name='Refl1dModel')


def wrapper_for(model: Model, resolution_function=None) -> Refl1dStatelessWrapper:
    wrapper = Refl1dStatelessWrapper()
    if resolution_function is not None:
        wrapper.set_resolution_function(resolution_function)
    wrapper.set_model(model)
    return wrapper


class TestSampleTraversal:
    def test_the_stack_is_built_from_the_substrate_up(self, clear):
        # Given, refl1d orders layers the opposite way to the sample
        model = build_model()

        # When
        sample = Refl1dStatelessWrapper()._build_sample(model)

        # Then
        assert [slab.name for slab in sample] == [
            model.sample[2].layers[0].unique_name,
            model.sample[1].layers[1].unique_name,
            model.sample[1].layers[0].unique_name,
            model.sample[0].layers[0].unique_name,
        ]

    def test_single_repetition_assemblies_are_flattened(self, clear):
        # Given
        model = build_model(repetitions=1.0)

        # When
        sample = Refl1dStatelessWrapper()._build_sample(model)

        # Then
        assert all(isinstance(component, names.Slab) for component in sample)

    def test_repeated_assemblies_become_a_repeat(self, clear):
        # Given
        model = build_model(repetitions=5.0)

        # When
        sample = Refl1dStatelessWrapper()._build_sample(model)

        # Then
        repeats = [component for component in sample if isinstance(component, Repeat)]
        assert len(repeats) == 1
        assert repeats[0].repeat.value == 5.0

    def test_material_values_are_read_from_the_model(self, clear):
        # Given
        model = build_model()

        # When
        sample = Refl1dStatelessWrapper()._build_sample(model)

        # Then, substrate first
        assert sample[0].material.rho.value == 4.0
        assert sample[0].material.irho.value == 0.0

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
        stack = Refl1dStatelessWrapper()._build_sample(model)

        # Then the mixed sld is used, not either constituent's
        assert_almost_equal(stack[0].material.rho.value, 3.0)

    def test_values_are_read_at_evaluation_time(self, clear):
        # Given
        model = build_model()
        wrapper = wrapper_for(model)
        before = wrapper.calculate(Q, model.unique_name)

        # When the model changes, with no call into the calculator at all
        model.sample[1].layers[0].thickness.value = 55.0

        # Then the next evaluation sees it
        assert not np.array_equal(before, wrapper.calculate(Q, model.unique_name))

    def test_structural_edits_need_no_mirroring(self, clear):
        # Given
        model = build_model()
        wrapper = wrapper_for(model)
        assert len(list(wrapper._build_sample(model))) == 4

        # When a layer is added through the model API only
        model.sample[1].add_layer(Layer(Material(3.0, 0.0, 'Extra'), 20.0, 1.0, 'ExtraLayer'))

        # Then
        assert len(list(wrapper._build_sample(model))) == 5


class TestModelRegistry:
    def test_the_sole_model_is_the_default(self, clear):
        # Given
        model = build_model()
        wrapper = wrapper_for(model)

        # When/Then
        assert_array_equal(wrapper.calculate(Q), wrapper.calculate(Q, model.unique_name))

    def test_several_models_require_a_model_id(self, clear):
        # Given
        wrapper = Refl1dStatelessWrapper()
        wrapper.set_model(build_model())
        wrapper.set_model(build_model())

        # When/Then
        with pytest.raises(ValueError, match='pass `model_id`'):
            wrapper.calculate(Q)

    def test_no_model_attached_raises(self, clear):
        # When/Then
        with pytest.raises(ValueError, match='No model is attached'):
            Refl1dStatelessWrapper().calculate(Q)


class TestResolution:
    def test_the_probe_receives_sigma(self, clear):
        # Given, refl1d's probe.dQ is a sigma, and `smearing` returns sigma
        q_knots = np.linspace(0.001, 0.5, 10)
        fwhm_knots = 0.02 * q_knots + 0.001
        resolution_function = LinearSpline(q_knots, fwhm_knots)
        model = build_model()
        wrapper = wrapper_for(model, resolution_function)
        captured = {}
        real_probe = names.QProbe

        def spy(**kwargs):
            captured['dQ'] = np.asarray(kwargs['dQ'], dtype=float)
            return real_probe(**kwargs)

        names.QProbe = spy
        try:
            wrapper.calculate(Q, model.unique_name)
        finally:
            names.QProbe = real_probe

        # Then
        assert_almost_equal(captured['dQ'], resolution_function.smearing(Q))


class TestProbeHelpers:
    def test_oversampling_q(self):
        # When
        oversampling = _get_oversampling_q(
            q_array=np.linspace(1, 10, 10),
            dq_array=np.linspace(0.01, 0.1, 10),
            oversampling_factor=5,
        )

        # Then
        assert len(oversampling) == 50
        assert oversampling[0] == 0.965
        assert oversampling[-1] == 10.35

    def test_probe_takes_scale_and_background_from_the_model(self, clear):
        # Given
        q = np.linspace(1, 10, 10)
        dq = np.linspace(0.01, 0.1, 10)
        model = build_model(scale=10.0, background=20.0)

        # When
        probe = _get_probe(q_array=q, dq_array=dq, model=model)

        # Then
        assert all(probe.Q == q)
        assert all(probe.calc_Q == q)
        assert all(probe.dQ == dq)
        assert probe.intensity.value == 10
        assert probe.background.value == 20

    def test_probe_oversampling(self, clear):
        # Given
        q = np.linspace(1, 10, 10)
        dq = np.linspace(0.01, 0.1, 10)

        # When
        probe = _get_probe(q_array=q, dq_array=dq, model=build_model(), oversampling_factor=2)

        # Then
        assert len(probe.calc_Q) == len(q)
        assert len(probe.calc_Qo) == 2 * len(q)

    def test_polarized_probe(self, clear):
        # Given
        q = np.linspace(1, 10, 10)
        dq = np.linspace(0.01, 0.1, 10)
        model = build_model(scale=10.0, background=20.0)

        # When
        probe = _get_polarized_probe(q_array=q, dq_array=dq, model=model)

        # Then
        assert all(probe.Q == q)
        assert all(probe.dQ == dq)
        assert len(probe.xs) == 4
        assert probe.xs[1:4] == [None, None, None]
        assert probe.xs[0].intensity.value == 10
        assert probe.xs[0].background.value == 20

    def test_polarized_probe_oversampling(self, clear):
        # Given
        q = np.linspace(1, 10, 10)
        dq = np.linspace(0.01, 0.1, 10)

        # When
        probe = _get_polarized_probe(q_array=q, dq_array=dq, model=build_model(), oversampling_factor=2)

        # Then
        assert len(probe.xs[0].calc_Qo) == 2 * len(q)


class TestMagnetism:
    def test_the_flag_is_off_by_default(self, clear):
        assert Refl1dStatelessWrapper().magnetism is False

    def test_magnetic_slabs_carry_a_magnetism_object(self, clear):
        # Given
        wrapper = Refl1dStatelessWrapper()
        wrapper.magnetism = True
        model = build_model()

        # When
        sample = wrapper._build_sample(model)

        # Then
        assert all(slab.magnetism is not None for slab in sample)

    def test_non_magnetic_slabs_carry_none(self, clear):
        # Given
        model = build_model()

        # When
        sample = Refl1dStatelessWrapper()._build_sample(model)

        # Then
        assert all(slab.magnetism is None for slab in sample)


class TestResolutionFunctionConfiguration:
    def test_percentage_fwhm_is_the_default(self, clear):
        assert isinstance(Refl1dStatelessWrapper()._resolution_function, PercentageFwhm)
