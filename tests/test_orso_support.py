# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ORSO support update (ORSO_UPDATE_TASK.md):

reader hardening (banner discriminator, FWHM->sigma, nan policy, 1/nm Q),
model-language preservation (units, repeats, density materials),
single-file polarized import, the .ort/.orb exporter, and .orb reading.
"""

import os

import numpy as np
import pytest
from easyscience import global_object

import easyreflectometry
from easyreflectometry.data import DataSet1D
from easyreflectometry.data import PolarizedDataSet
from easyreflectometry.data.measurement import load
from easyreflectometry.data.measurement import load_as_dataset
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model import Pointwise
from easyreflectometry.orso_utils import SIGMA_TO_FWHM
from easyreflectometry.orso_utils import _load_orso_any
from easyreflectometry.orso_utils import is_orso_file
from easyreflectometry.orso_utils import load_orso_model
from easyreflectometry.orso_utils import sample_to_orso_model
from easyreflectometry.orso_utils import save_orso_experiment
from easyreflectometry.project import Project
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import RepeatingMultilayer
from easyreflectometry.sample import Sample
from easyreflectometry.sample.elements.materials.material_density import MaterialDensity

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')

# The Qz/sQz grids the generated fixtures were built from (see task 7a).
FIXTURE_QZ = np.linspace(0.01, 0.3, 12)
FIXTURE_SQZ = FIXTURE_QZ * 0.02


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


@pytest.fixture
def project() -> Project:
    return Project()


class TestBannerDiscriminator:
    def test_ort_file_is_orso(self):
        assert is_orso_file(os.path.join(PATH_STATIC, 'Ni_example.ort')) is True

    def test_orb_file_is_orso(self):
        assert is_orso_file(os.path.join(PATH_STATIC, 'example.orb')) is True

    def test_txt_file_is_not_orso(self):
        assert is_orso_file(os.path.join(PATH_STATIC, 'test_example1.txt')) is False

    def test_bannered_but_corrupt_file_raises(self, tmp_path):
        # A file carrying the ORSO banner that fails to parse must raise, not
        # silently fall back to plain-text loading (which drops the header).
        bad = tmp_path / 'bad.ort'
        bad.write_text(
            '# # ORSO reflectivity data file | 1.1 standard | YAML encoding | https://www.reflectometry.org/\n'
            '# data_source: {[[not yaml\n'
            '1.0 2.0 3.0 4.0\n'
        )
        with pytest.raises(ValueError, match='Error loading ORSO file'):
            load(str(bad))

    def test_non_bannered_file_falls_back_to_txt(self, tmp_path):
        plain = tmp_path / 'plain.dat'
        plain.write_text('0.01 1.0 0.1 0.001\n0.02 0.5 0.05 0.002\n')
        data_group = load(str(plain))
        assert 'R_plain' in data_group['data']

    def test_count_datasets_raises_for_corrupt_orso(self, project, tmp_path):
        bad = tmp_path / 'bad.ort'
        bad.write_text('# # ORSO reflectivity data file | 1.1 standard | YAML encoding\n# data_source: {[[\n1 2 3 4\n')
        with pytest.raises(ValueError):
            project.count_datasets_in_file(str(bad))

    def test_count_datasets_multi(self, project):
        assert project.count_datasets_in_file(os.path.join(PATH_STATIC, 'polarized_2ch.ort')) == 2


class TestNanPolicy:
    def test_all_nan_sqz_leaves_xe_empty(self):
        data_group = load(os.path.join(PATH_STATIC, 'nan_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        assert coords.variances is None

    def test_all_nan_sqz_falls_back_to_percentage_fwhm(self, project):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'nan_sqz.ort'))
        assert isinstance(project.models[0].resolution_function, PercentageFwhm)

    def test_partial_nan_sqz_is_interpolated(self):
        with pytest.warns(UserWarning, match='sQz values are nan'):
            data_group = load(os.path.join(PATH_STATIC, 'partial_nan_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        assert coords.variances is not None
        assert np.all(np.isfinite(coords.variances))

    def test_partial_nan_sqz_builds_finite_pointwise(self, project):
        project.default_model()
        with pytest.warns(UserWarning, match='sQz values are nan'):
            project.load_new_experiment(os.path.join(PATH_STATIC, 'partial_nan_sqz.ort'))
        resolution_function = project.models[0].resolution_function
        assert isinstance(resolution_function, Pointwise)
        assert np.all(np.isfinite(resolution_function.smearing(FIXTURE_QZ)))


class TestValueIsFwhm:
    def test_fwhm_sqz_is_converted_to_sigma(self):
        data_group = load(os.path.join(PATH_STATIC, 'fwhm_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        # The fixture stores sigma * SIGMA_TO_FWHM declared as FWHM; loading
        # must convert back so stored variances are sigma squared.
        np.testing.assert_allclose(np.sqrt(coords.variances), FIXTURE_SQZ)

    def test_sigma_file_is_not_scaled(self):
        data_group = load(os.path.join(PATH_STATIC, 'partial_nan_sqz.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        valid = np.sqrt(coords.variances)[2:-1]
        np.testing.assert_allclose(valid, FIXTURE_SQZ[2:-1])


class TestQUnitConversion:
    def test_1_per_nm_qz_is_converted_to_1_per_angstrom(self):
        data_group = load(os.path.join(PATH_STATIC, 'nm_units.ort'))
        coords = data_group['coords'][list(data_group['coords'])[0]]
        np.testing.assert_allclose(coords.values, FIXTURE_QZ)
        np.testing.assert_allclose(np.sqrt(coords.variances), FIXTURE_SQZ)
        assert str(coords.unit) == '1/Å'


class TestModelLanguagePreservation:
    def test_nm_default_length_unit_is_honoured(self):
        # nm_units.ort declares thicknesses as bare magnitudes with no
        # length_unit -> the model-language default (nm) applies.
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'nm_units.ort')))
        film = [layer for assembly in sample for layer in assembly.layers if layer.name == 'film'][0]
        assert film.thickness.value == pytest.approx(100.0)  # 10 nm
        assert film.roughness.value == pytest.approx(5.0)  # 0.5 nm

    def test_default_roughness_from_globals(self):
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'nm_units.ort')))
        subphase_layer = sample[-1].layers[0]
        # Si declares roughness 0.3 (nm default) -> 3 A
        assert subphase_layer.roughness.value == pytest.approx(3.0)

    def test_repeated_substack_becomes_repeating_multilayer(self):
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'nm_units.ort')))
        repeating = [assembly for assembly in sample if isinstance(assembly, RepeatingMultilayer)]
        assert len(repeating) == 1
        assert repeating[0].repetitions.value == 3
        assert [layer.name for layer in repeating[0].layers] == ['A', 'B']
        assert repeating[0].layers[0].thickness.value == pytest.approx(20.0)  # 2 nm

    def test_density_material_stays_density_defined(self):
        sample = load_orso_model(_load_orso_any(os.path.join(PATH_STATIC, 'Ni_example.ort')))
        m1 = sample[1].layers[0]
        assert isinstance(m1.material, MaterialDensity)
        assert m1.material.chemical_structure == 'Ni'
        assert m1.material.density.value == pytest.approx(8.9)
        # and the derived SLD is still sensible (Ni ~ 9.4e-6 A^-2)
        assert m1.material.sld.value == pytest.approx(9.4, abs=0.1)


class TestPolarizedSingleFile:
    def test_load_polarized_experiment_from_file(self, project):
        project.default_model()
        index = project.load_polarized_experiment_from_file(os.path.join(PATH_STATIC, 'polarized_2ch.ort'))
        experiment = project.experiments[index]
        assert isinstance(experiment, PolarizedDataSet)
        assert [channel.value for channel in experiment.available_channels] == ['pp', 'mm']
        assert experiment.name == 'Polarized fixture'
        assert experiment.model is project.models[0]
        np.testing.assert_allclose(experiment['pp'].x, FIXTURE_QZ)

    def test_unclassifiable_dataset_raises(self, project):
        project.default_model()
        # nan_sqz.ort carries no spin-channel polarization -> not coerced.
        with pytest.raises(ValueError, match='declares no (mappable spin channel|polarization)'):
            project.load_polarized_experiment_from_file(os.path.join(PATH_STATIC, 'nan_sqz.ort'))

    def test_multidataset_file_rejected_by_per_channel_loader(self, project):
        project.default_model()
        with pytest.raises(ValueError, match='load_polarized_experiment_from_file'):
            project.load_polarized_experiment({'pp': os.path.join(PATH_STATIC, 'polarized_2ch.ort')})

    def test_duplicate_channel_raises(self, project, tmp_path):
        # Build a file where two datasets declare the same channel.
        datasets = _load_orso_any(os.path.join(PATH_STATIC, 'polarized_2ch.ort'))
        from orsopy.fileio.data_source import Polarization
        from orsopy.fileio.orso import save_orso

        for orso_dataset in datasets:
            orso_dataset.info.data_source.measurement.instrument_settings.polarization = Polarization('pp')
        duplicate_file = tmp_path / 'dup.ort'
        save_orso(datasets, str(duplicate_file))

        project.default_model()
        with pytest.raises(ValueError, match="Duplicate spin channel 'pp'"):
            project.load_polarized_experiment_from_file(str(duplicate_file))


class TestExporter:
    def test_ort_data_roundtrip_writes_sigma(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        experiment = project.experiments[0]
        out = tmp_path / 'out.ort'
        project.save_experiment_as_orso(str(out), 0)

        back = _load_orso_any(str(out))
        assert len(back) == 1
        data = back[0].data
        np.testing.assert_allclose(data[:, 0], experiment.x)
        np.testing.assert_allclose(data[:, 1], experiment.y)
        # DataSet1D stores variances; the file must carry sigma.
        np.testing.assert_allclose(data[:, 2], np.sqrt(experiment.ye))
        np.testing.assert_allclose(data[:, 3], np.sqrt(experiment.xe))
        # sigma convention declared in the columns
        assert back[0].info.columns[2].value_is == 'sigma'
        assert back[0].info.columns[3].value_is == 'sigma'

    def test_export_reuses_preserved_header(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        out = tmp_path / 'out.ort'
        project.save_experiment_as_orso(str(out), 0)
        info = _load_orso_any(str(out))[0].info
        # data_source/reduction provenance from the original file, not synthesized
        assert info.data_source.experiment.title == 'Metal films'
        assert info.data_source.owner.name == 'Joe Bloggs'

    def test_export_writes_model_language(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        out = tmp_path / 'out.ort'
        project.save_experiment_as_orso(str(out), 0)
        model = _load_orso_any(str(out))[0].info.data_source.sample.model
        assert model is not None
        assert model.globals.length_unit == 'angstrom'
        # the exported model resolves back into layers
        assert len(model.resolve_to_layers()) >= 2

    def test_absent_errors_written_as_nan(self, tmp_path):
        dataset = DataSet1D(x=np.array([0.01, 0.02]), y=np.array([1.0, 0.5]))
        out = tmp_path / 'nan.ort'
        save_orso_experiment(dataset, str(out))
        data = _load_orso_any(str(out))[0].data
        assert np.all(np.isnan(data[:, 2]))
        assert np.all(np.isnan(data[:, 3]))

    def test_polarized_export_single_multidataset_file(self, project, tmp_path):
        project.default_model()
        index = project.load_polarized_experiment_from_file(os.path.join(PATH_STATIC, 'polarized_2ch.ort'))
        out = tmp_path / 'pol.ort'
        project.save_experiment_as_orso(str(out), index)

        back = _load_orso_any(str(out))
        assert [d.info.data_set for d in back] == ['pp', 'mm']
        polarizations = [str(d.info.data_source.measurement.instrument_settings.polarization.value) for d in back]
        assert polarizations == ['pp', 'mm']

    def test_repeating_multilayer_roundtrips_via_model_language(self):
        air = Layer(material=Material(sld=0.0, isld=0.0, name='air'), thickness=0, roughness=0, name='air')
        layer_a = Layer(material=Material(sld=4.0, isld=0.0, name='A'), thickness=20, roughness=3, name='A')
        layer_b = Layer(material=Material(sld=2.0, isld=0.0, name='B'), thickness=10, roughness=3, name='B')
        si = Layer(material=Material(sld=2.07, isld=0.0, name='Si'), thickness=0, roughness=3, name='Si')
        sample = Sample(
            Multilayer(air, name='Superphase'),
            RepeatingMultilayer([layer_a, layer_b], repetitions=5, name='rep'),
            Multilayer(si, name='Subphase'),
            name='test',
        )
        orso_model = sample_to_orso_model(sample)
        assert '5 ( A | B )' in orso_model.stack
        assert len(orso_model.resolve_to_layers()) == 12  # 1 + 5*2 + 1

    def test_model_as_orso_returns_model_language_dict(self, project):
        project.default_model()
        orso_dict = project.models[0].as_orso()
        assert 'stack' in orso_dict
        assert 'layers' in orso_dict
        assert orso_dict['globals']['length_unit'] == 'angstrom'


class TestOrbSupport:
    def test_load_orb_file(self):
        data_group = load(os.path.join(PATH_STATIC, 'example.orb'))
        assert 'R_0' in data_group['data']
        coords = data_group['coords'][list(data_group['coords'])[0]]
        np.testing.assert_allclose(coords.values, FIXTURE_QZ)

    def test_orb_write_and_read_roundtrip(self, project, tmp_path):
        project.default_model()
        project.load_new_experiment(os.path.join(PATH_STATIC, 'Ni_example.ort'))
        experiment = project.experiments[0]
        out = tmp_path / 'out.orb'
        project.save_experiment_as_orso(str(out), 0)
        back = _load_orso_any(str(out))
        np.testing.assert_allclose(back[0].data[:, 0], experiment.x)
        np.testing.assert_allclose(back[0].data[:, 2], np.sqrt(experiment.ye))

    def test_load_orb_as_dataset(self):
        dataset = load_as_dataset(os.path.join(PATH_STATIC, 'example.orb'))
        assert isinstance(dataset, DataSet1D)
        np.testing.assert_allclose(dataset.x, FIXTURE_QZ)
        assert dataset.orso_header is not None


class TestFwhmSigmaContract:
    def test_pointwise_serialization_stays_variances(self):
        # The saved-project contract: sQz_data_points round-trip as variances.
        qz = np.array([0.01, 0.02])
        reflectivity = np.array([1.0, 0.5])
        variances = np.array([1e-8, 2e-8])
        pointwise = Pointwise([qz, reflectivity, variances])
        as_dict = pointwise.as_dict()
        np.testing.assert_allclose(as_dict['sQz_data_points'], variances)
        from easyreflectometry.model.resolution_functions import ResolutionFunction

        restored = ResolutionFunction.from_dict(as_dict)
        np.testing.assert_allclose(restored.smearing(qz), np.sqrt(variances))

    def test_sigma_to_fwhm_constant(self):
        assert SIGMA_TO_FWHM == pytest.approx(2.3548, abs=1e-4)
