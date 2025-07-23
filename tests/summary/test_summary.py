import os
from unittest.mock import MagicMock

import pytest
from easyscience import global_object

import easyreflectometry
from easyreflectometry import Project
from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.summary import Summary

PATH_STATIC = os.path.join(os.path.dirname(easyreflectometry.__file__), '..', '..', 'tests', '_static')


class TestSummary:
    @pytest.fixture
    def project(self) -> Project:
        global_object.map._clear()
        project = Project()
        project.default_model()
        return project

    def test_constructor(self, project: Project) -> None:
        # When Then
        result = Summary(project)

        # Expect
        assert result._project == project

    def test_compile_html_summary(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary._project_information_section = MagicMock(return_value='project result html')
        summary._sample_section = MagicMock(return_value='sample result html')
        summary._experiments_section = MagicMock(return_value='experiments results html')
        summary._refinement_section = MagicMock(return_value='refinement result html')
        summary._figures_section = MagicMock()

        # Then
        result = summary.compile_html_summary()

        # Expect
        summary._figures_section.assert_not_called()
        assert 'project result html' in result
        assert 'sample result html' in result
        assert 'experiments results html' in result
        assert 'refinement result html' in result
        assert 'figures_section' not in result

    def test_compile_html_summary_with_figures(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary._project_information_section = MagicMock(return_value='project result html')
        summary._sample_section = MagicMock(return_value='sample result html')
        summary._experiments_section = MagicMock(return_value='experiments results html')
        summary._refinement_section = MagicMock(return_value='refinement result html')
        summary._figures_section = MagicMock(return_value='figures result html')

        # Then
        result = summary.compile_html_summary(figures=True)

        # Expect
        assert 'figures result html' in result

    def test_save_html_summary(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        summary.compile_html_summary = MagicMock(return_value='html')
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.html')

        # Then
        summary.save_html_summary(file_path)

        # Expect
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            assert f.read() == 'html'

    def test_save_pdf_summary(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        summary.compile_html_summary = MagicMock(return_value='html')
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.pdf')

        # Then
        summary.save_pdf_summary(file_path)

        # Expect
        assert os.path.exists(file_path)

    def test_project_information_section(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then
        html = summary._project_information_section()

        # Expect
        assert 'DefaultEasyReflectometryProject' in html
        assert 'Reflectometry, 1D' in html

    def test_sample_section(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then
        html = summary._sample_section()

        # Expect
        assert 'Name' in html
        assert 'Value' in html
        assert 'Unit' in html
        assert 'Error' in html

        assert 'sld' in html
        assert 'isld' in html
        assert 'thickness' in html
        assert 'background' in html

    def test_experiments_section(self, project: Project) -> None:
        # When
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)
        summary = Summary(project)

        # Then
        html = summary._experiments_section()

        # Expect
        assert 'Experiment for Model 0' in html
        assert 'No. of data points' in html
        assert '408' in html
        assert 'Resolution function' in html
        assert 'Pointwise' in html

    def test_experiments_section_percentage_fhwm(self, project: Project) -> None:
        # When
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)
        project.models[0].resolution_function = PercentageFwhm(5)
        summary = Summary(project)

        # Then
        html = summary._experiments_section()

        # Expect
        assert 'PercentageFwhm 5%' in html

    def test_refinement_section(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then
        html = summary._refinement_section()

        # Expect
        assert 'refnx' in html
        assert 'LMFit_leastsq' in html
        assert 'No. of parameters:' in html
        assert 'No. of fixed parameters:' in html
        assert '14' in html
        assert 'No. of free parameters:' in html
        assert '0' in html
        assert 'No. of constraints' in html

    def test_save_sld_plot(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.jpg')

        # Then
        summary.save_sld_plot(file_path)

        # Expect
        assert os.path.exists(file_path)

    def test_save_fit_experiment_plot(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.jpg')
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)

        # Then
        summary.save_fit_experiment_plot(file_path)

        # Expect
        assert os.path.exists(file_path)

    def test_figures_section(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary.save_sld_plot = MagicMock()
        summary.save_fit_experiment_plot = MagicMock()

        # Then
        html = summary._figures_section()

        # Expect
        summary.save_sld_plot.assert_called_once()
        summary.save_fit_experiment_plot.assert_called_once()
        assert 'sld_plot' in html
        assert 'fit_experiment_plot' in html
