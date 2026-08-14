# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for explicit-channel calculation, the polarized reflectivity cache,
per-channel experiment loading, and simultaneous multi-channel fitting.
"""

from unittest.mock import patch

import numpy as np
import pytest
from easyscience import global_object
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.calculators import PolarizationChannel
from easyreflectometry.calculators.refl1d import wrapper as refl1d_wrapper
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.fitting import MultiFitter
from easyreflectometry.model import Model
from easyreflectometry.model import ModelCollection
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import LayerMagnetism
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample

Q = np.linspace(0.005, 0.3, 50)


def _magnetic_model(magnetism: LayerMagnetism | None) -> Model:
    vacuum = Material(sld=0, isld=0, name='Vacuum')
    material = Material(sld=4.0, isld=0, name='Sld 4')
    si = Material(sld=2.047, isld=0, name='Si')
    superphase = Layer(material=vacuum, thickness=0, roughness=0, name='Vacuum Superphase')
    layer = Layer(material=material, thickness=100, roughness=0, magnetism=magnetism, name='Sld 4 Layer')
    subphase = Layer(material=si, thickness=0, roughness=0, name='Si Subphase')
    sample = Sample(Multilayer(superphase), Multilayer(layer), Multilayer(subphase), name='Sample')
    model = Model(sample=sample, scale=1, background=0, name='Magnetic Model')
    model.resolution_function = PercentageFwhm(0)
    return model


def _refl1d_interface() -> CalculatorFactory:
    interface = CalculatorFactory()
    interface.switch('refl1d')
    return interface


def _polarized_data(channels: dict[str, np.ndarray], model=None) -> PolarizedDataSet:
    datasets = {
        channel: DataSet1D(name=channel, x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)
        for channel, reflectivity in channels.items()
    }
    return PolarizedDataSet(name='synthetic', channels=datasets, model=model)


class TestCalculateChannel:
    def test_channels_match_calculate_polarized(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()

        reference = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
        for channel in ('pp', 'pm', 'mp', 'mm'):
            assert_allclose(
                calculator.reflectivity_profile_channel(Q, model.unique_name, channel),
                reference[channel],
                rtol=1e-12,
            )

    def test_pp_without_magnetism_falls_back_to_unpolarized(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        calculator = model.interface()

        assert_allclose(
            calculator.reflectivity_profile_channel(Q, model.unique_name, 'pp'),
            calculator.reflectity_profile(Q, model.unique_name),
            rtol=1e-12,
        )

    def test_spin_flip_without_magnetism_raises(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        with pytest.raises(ValueError):
            model.interface().reflectivity_profile_channel(Q, model.unique_name, 'pm')

    def test_fit_func_for_channel(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()

        reference = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)
        fit_func = model.interface.fit_func_for_channel('mm')
        assert_allclose(fit_func(Q, model.unique_name), reference['mm'], rtol=1e-12)


class TestPolarizedCache:
    def test_repeated_calculation_hits_cache(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            first = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            second = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            assert experiment.call_count == 1
            # All four channels through calculate_channel: still no new evaluation.
            for channel in ('pp', 'pm', 'mp', 'mm'):
                calculator.reflectivity_profile_channel(Q, model.unique_name, channel)
            assert experiment.call_count == 1
        for channel in ('pp', 'pm', 'mp', 'mm'):
            assert_allclose(first[channel], second[channel], rtol=1e-15)

    def test_parameter_change_invalidates_cache(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            before = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            model.sample[1].layers[0].magnetism.rho_m = 3.0
            after = calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            assert experiment.call_count == 2
        assert not np.allclose(before['mm'], after['mm'])

    def test_q_dtype_is_normalized_before_keying(self):
        # Keying happens after normalization to float64 plus explicit shape, so
        # byte-identical arrays of different dtype/shape can never collide. A
        # float32 grid whose values are exactly representable normalizes to the
        # same key as its float64 twin and shares the cache entry.
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()
        q_pow2 = np.array([0.03125, 0.0625, 0.125, 0.25])  # exact in float32

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            first = calculator.polarized_reflectivity_profiles(q_pow2, model.unique_name)
            second = calculator.polarized_reflectivity_profiles(q_pow2.astype(np.float32), model.unique_name)
            assert experiment.call_count == 1
        for channel in ('pp', 'pm', 'mp', 'mm'):
            assert len(first[channel]) == len(q_pow2)
            assert_allclose(second[channel], first[channel], rtol=1e-15)

    def test_different_q_grids_coexist_within_one_state(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        calculator = model.interface()
        q_other = np.linspace(0.01, 0.2, 30)

        with patch.object(refl1d_wrapper.names, 'Experiment', wraps=refl1d_wrapper.names.Experiment) as experiment:
            calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            calculator.polarized_reflectivity_profiles(q_other, model.unique_name)
            assert experiment.call_count == 2
            # Both grids now cached for the same model state.
            calculator.polarized_reflectivity_profiles(Q, model.unique_name)
            calculator.polarized_reflectivity_profiles(q_other, model.unique_name)
            assert experiment.call_count == 2


class TestLoadPolarizedExperiment:
    @staticmethod
    def _write_channel_file(directory, name: str) -> str:
        path = directory / name
        q = np.linspace(0.01, 0.2, 20)
        reflectivity = np.exp(-q * 30)
        error = 0.01 * reflectivity
        np.savetxt(path, np.column_stack([q, reflectivity, error]))
        return str(path)

    def test_load_polarized_experiment(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        new_index = project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})

        # The index is returned so a GUI can make the new experiment current.
        assert new_index == 0
        experiment = project.experiments[0]
        assert isinstance(experiment, PolarizedDataSet)
        assert experiment.available_channels == [PolarizationChannel.PP, PolarizationChannel.MM]
        assert experiment.name == 'Polarized experiment 0'
        assert experiment.model is project.models[0]
        assert experiment['pp'].model is project.models[0]
        assert len(experiment['pp'].x) == 20

    def test_second_polarized_experiment_gets_the_next_index(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        first = project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})
        second = project.load_polarized_experiment({'pp': pp_path})

        assert (first, second) == (0, 1)
        assert len(project.experiments) == 2

    def test_multi_dataset_file_is_rejected(self, tmp_path):
        import os

        multi_path = os.path.join(os.path.dirname(__file__), '_static', 'test_example2.ort')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')

        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        with pytest.raises(ValueError, match='multiple datasets'):
            project.load_polarized_experiment({'pp': multi_path, 'mm': mm_path})
        assert len(project.experiments) == 0

    def test_suggest_polarized_channel_assignment(self, tmp_path):
        pp_path = self._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = self._write_channel_file(tmp_path, 'sample_dd.txt')
        unknown_path = self._write_channel_file(tmp_path, 'sample_other.txt')

        project = Project()
        suggestion = project.suggest_polarized_channel_assignment([pp_path, mm_path, unknown_path])

        assert suggestion[str(pp_path)] == PolarizationChannel.PP
        assert suggestion[str(mm_path)] == PolarizationChannel.MM
        assert suggestion[str(unknown_path)] is None


class TestChannelAwareExperimentAccessors:
    """`experimental_data_for_model_at_index(index, channel=…)` and friends."""

    @staticmethod
    def _polarized_project(tmp_path) -> Project:
        pp_path = TestLoadPolarizedExperiment._write_channel_file(tmp_path, 'sample_uu.txt')
        mm_path = TestLoadPolarizedExperiment._write_channel_file(tmp_path, 'sample_dd.txt')
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        project.load_polarized_experiment({'pp': pp_path, 'mm': mm_path})
        return project

    def test_without_channel_returns_the_whole_experiment(self, tmp_path):
        project = self._polarized_project(tmp_path)

        experiment = project.experimental_data_for_model_at_index(0)

        assert isinstance(experiment, PolarizedDataSet)
        assert project.experiment_is_polarized_at_index(0) is True
        assert project.experiment_channels_at_index(0) == [PolarizationChannel.PP, PolarizationChannel.MM]

    def test_channel_returns_that_channel_dataset(self, tmp_path):
        project = self._polarized_project(tmp_path)
        experiment = project.experiments[0]

        for channel in ('pp', PolarizationChannel.MM):
            data = project.experimental_data_for_model_at_index(0, channel=channel)
            assert isinstance(data, DataSet1D)
            assert data is experiment[channel]

    def test_unmeasured_channel_raises_key_error(self, tmp_path):
        project = self._polarized_project(tmp_path)

        with pytest.raises(KeyError, match='was not measured'):
            project.experimental_data_for_model_at_index(0, channel='pm')

    def test_unknown_channel_raises_value_error(self, tmp_path):
        project = self._polarized_project(tmp_path)

        with pytest.raises(ValueError, match='Unknown spin channel'):
            project.experimental_data_for_model_at_index(0, channel='xx')

    def test_channel_on_unpolarized_experiment_raises_value_error(self, tmp_path):
        path = TestLoadPolarizedExperiment._write_channel_file(tmp_path, 'sample.txt')
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()
        project.load_experiment_for_model_at_index(path, 0)

        assert project.experiment_is_polarized_at_index(0) is False
        assert project.experiment_channels_at_index(0) == []
        with pytest.raises(ValueError, match='not polarized'):
            project.experimental_data_for_model_at_index(0, channel='pp')

    def test_missing_experiment_raises_index_error(self):
        project = Project()
        project.default_model()

        assert project.experiment_is_polarized_at_index(0) is False
        with pytest.raises(IndexError):
            project.experimental_data_for_model_at_index(0, channel='pp')

    def test_model_data_per_channel_differs_for_magnetic_model(self):
        global_object.map._clear()
        model = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=40.0))
        project = Project()
        project.calculator = 'refl1d'
        project.models = ModelCollection(model)
        q_range = np.linspace(0.01, 0.2, 25)

        pp = project.model_data_for_model_at_index(0, q_range=q_range, channel='pp')
        mm = project.model_data_for_model_at_index(0, q_range=q_range, channel='mm')
        pm = project.model_data_for_model_at_index(0, q_range=q_range, channel='pm')

        assert pp.name.endswith('(pp) for Model 0')
        # Each cross-section is genuinely different — this is what a per-channel
        # display/report must show instead of one curve repeated four times.
        assert not np.allclose(pp.y, mm.y)
        assert not np.allclose(pp.y, pm.y)

    def test_spin_flip_channel_of_non_magnetic_model_raises(self):
        global_object.map._clear()
        project = Project()
        project.calculator = 'refl1d'
        project.default_model()

        with pytest.raises(ValueError, match='requires magnetism'):
            project.model_data_for_model_at_index(0, channel='pm')


class TestFitPolarized:
    def test_two_channel_nsf_fit_recovers_rho_m(self):
        truth = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=1.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        rho_m = model.sample[1].layers[0].magnetism.rho_m
        rho_m.fixed = False
        rho_m.bounds = (0.0, 5.0)

        data = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=model)
        fitter = MultiFitter(model)
        results = fitter.fit_polarized(data)

        assert list(results.keys()) == ['pp', 'mm']
        assert all(result.success for result in results.values())
        assert_allclose(rho_m.value, 2.5, atol=0.01)

    def test_four_channel_fit_recovers_rho_m_and_theta_m(self):
        truth = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=45.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=1.5, theta_m=60.0))
        model.interface = _refl1d_interface()
        magnetism = model.sample[1].layers[0].magnetism
        magnetism.rho_m.fixed = False
        magnetism.rho_m.bounds = (0.0, 5.0)
        magnetism.theta_m.fixed = False
        magnetism.theta_m.bounds = (0.0, 90.0)

        data = _polarized_data(dict(reference), model=model)
        fitter = MultiFitter(model)
        results = fitter.fit_polarized(data)

        assert list(results.keys()) == ['pp', 'pm', 'mp', 'mm']
        assert all(result.success for result in results.values())
        assert_allclose(magnetism.rho_m.value, 2.5, atol=0.02)
        assert_allclose(magnetism.theta_m.value, 45.0, atol=0.5)

    def test_shared_structural_parameter_fitted_across_channels(self):
        truth = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        thickness = model.sample[1].layers[0].thickness
        thickness.value = 90.0
        thickness.fixed = False
        thickness.bounds = (50.0, 150.0)

        data = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=model)
        fitter = MultiFitter(model)
        results = fitter.fit_polarized(data)

        assert all(result.success for result in results.values())
        assert_allclose(thickness.value, 100.0, atol=0.1)

    def test_fit_polarized_requires_matching_model(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        other = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        other.interface = _refl1d_interface()

        data = _polarized_data({'pp': np.ones_like(Q)}, model=other)
        fitter = MultiFitter(model)
        with pytest.raises(ValueError, match='must be the model'):
            fitter.fit_polarized(data)

    def test_fit_polarized_requires_matching_channel_models(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        other = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        other.interface = _refl1d_interface()

        data = _polarized_data({'pp': np.ones_like(Q), 'mm': np.ones_like(Q)}, model=model)
        # Rebind one channel dataset behind the experiment's back.
        data['mm'].model = other

        fitter = MultiFitter(model)
        with pytest.raises(ValueError, match="'mm' channel dataset"):
            fitter.fit_polarized(data)

    def test_fit_polarized_requires_single_model(self):
        model_a = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        model_b = _magnetic_model(None)
        interface = _refl1d_interface()
        model_a.interface = interface
        model_b.interface = interface

        data = _polarized_data({'pp': np.ones_like(Q)}, model=model_a)
        fitter = MultiFitter(model_a, model_b)
        with pytest.raises(ValueError):
            fitter.fit_polarized(data)


class TestMultiFitterForExperiments:
    """`MultiFitter.for_experiments` — one fit function per dataset, channels expanded."""

    @staticmethod
    def _unpolarized_data(model, name='plain') -> DataSet1D:
        reflectivity = np.exp(-Q * 30)
        dataset = DataSet1D(name=name, x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)
        dataset.model = model
        return dataset

    def test_polarized_experiment_expands_to_one_function_per_channel(self):
        model = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=45.0))
        model.interface = _refl1d_interface()
        reference = model.interface.polarized_reflectivity_profiles(Q, model.unique_name)
        data = _polarized_data(dict(reference), model=model)

        fitter = MultiFitter.for_experiments([data])

        assert fitter.fit_channels == [
            PolarizationChannel.PP,
            PolarizationChannel.PM,
            PolarizationChannel.MP,
            PolarizationChannel.MM,
        ]
        assert fitter.fit_datasets == [data[channel] for channel in data.available_channels]
        assert len(fitter._fit_func) == 4
        # Each function evaluates its own cross-section, not four copies of one.
        curves = [func(Q) for func in fitter._fit_func]
        for index, channel in enumerate(data.available_channels):
            assert_allclose(curves[index], reference[channel.value], rtol=1e-9)

    def test_unpolarized_experiment_keeps_one_function(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        data = self._unpolarized_data(model)

        fitter = MultiFitter.for_experiments([data])

        assert fitter.fit_channels == [None]
        assert fitter.fit_datasets == [data]
        assert_allclose(fitter._fit_func[0](Q), model.interface.fit_func(Q, model.unique_name), rtol=1e-9)

    def test_mixed_experiments_share_one_fitter(self):
        """A polarized and an ordinary experiment fitted together, two models."""
        magnetic = _magnetic_model(LayerMagnetism(rho_m=2.0, theta_m=270.0))
        plain = _magnetic_model(None)
        interface = _refl1d_interface()
        magnetic.interface = interface
        plain.interface = interface
        reference = magnetic.interface.polarized_reflectivity_profiles(Q, magnetic.unique_name)
        polarized = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=magnetic)
        unpolarized = self._unpolarized_data(plain)

        fitter = MultiFitter.for_experiments([polarized, unpolarized])

        assert fitter.fit_channels == [PolarizationChannel.PP, PolarizationChannel.MM, None]
        # Both models' parameters are enumerated, so they are fitted together.
        assert len(fitter._models) == 2
        assert len(fitter.easy_science_multi_fitter._fit_functions) == 3

    def test_repeated_model_is_registered_once(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        first = self._unpolarized_data(model, name='a')
        second = self._unpolarized_data(model, name='b')

        fitter = MultiFitter.for_experiments([first, second])

        assert len(fitter._models) == 1
        assert len(fitter.fit_datasets) == 2

    def test_experiment_without_model_is_rejected(self):
        dataset = DataSet1D(name='orphan', x=Q, y=np.ones_like(Q), ye=np.ones_like(Q))
        dataset.model = None

        with pytest.raises(ValueError, match='no model'):
            MultiFitter.for_experiments([dataset])

    def test_empty_experiment_list_is_rejected(self):
        with pytest.raises(ValueError, match='At least one experiment'):
            MultiFitter.for_experiments([])

    def test_prepared_fitter_recovers_rho_m_when_run(self):
        """The fitter is usable exactly like `fit_polarized`, but caller-driven."""
        truth = _magnetic_model(LayerMagnetism(rho_m=2.5, theta_m=270.0))
        truth.interface = _refl1d_interface()
        reference = truth.interface.polarized_reflectivity_profiles(Q, truth.unique_name)

        model = _magnetic_model(LayerMagnetism(rho_m=1.0, theta_m=270.0))
        model.interface = _refl1d_interface()
        rho_m = model.sample[1].layers[0].magnetism.rho_m
        rho_m.fixed = False
        rho_m.bounds = (0.0, 5.0)
        data = _polarized_data({'pp': reference['pp'], 'mm': reference['mm']}, model=model)

        fitter = MultiFitter.for_experiments([data])
        x = [np.asarray(dataset.x) for dataset in fitter.fit_datasets]
        y = [np.asarray(dataset.y) for dataset in fitter.fit_datasets]
        weights = [1.0 / np.sqrt(np.asarray(dataset.ye)) for dataset in fitter.fit_datasets]
        results = fitter.easy_science_multi_fitter.fit(x, y, weights=weights)

        assert all(result.success for result in results)
        assert_allclose(rho_m.value, 2.5, atol=0.01)


class TestRecordFitResults:
    """Results produced by a caller-driven fit can be handed back to the fitter."""

    def _fitter_and_results(self):
        model = _magnetic_model(None)
        model.interface = _refl1d_interface()
        reflectivity = np.exp(-Q * 30)
        dataset = DataSet1D(name='plain', x=Q, y=reflectivity, ye=(0.01 * reflectivity) ** 2)
        dataset.model = model
        fitter = MultiFitter.for_experiments([dataset])
        results = fitter.easy_science_multi_fitter.fit(
            [np.asarray(dataset.x)], [np.asarray(dataset.y)], weights=[1.0 / np.sqrt(np.asarray(dataset.ye))]
        )
        return fitter, list(results)

    def test_metrics_are_none_before_recording(self):
        fitter, _results = self._fitter_and_results()

        # `easy_science_multi_fitter.fit` bypasses MultiFitter entirely.
        assert fitter.chi2 is None
        assert fitter.reduced_chi is None

    def test_recording_makes_the_metrics_available(self):
        fitter, results = self._fitter_and_results()

        fitter.record_fit_results(results)

        assert fitter.chi2 == pytest.approx(sum(r.chi2 for r in results))
        assert fitter.reduced_chi is not None
        # The classical metrics need the original arrays, which FitResults lacks.
        assert fitter.classical_chi2 is None

    def test_recording_none_clears_the_metrics(self):
        fitter, results = self._fitter_and_results()
        fitter.record_fit_results(results)

        fitter.record_fit_results(None)

        assert fitter.chi2 is None
