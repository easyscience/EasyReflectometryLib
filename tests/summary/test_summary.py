# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import os
from unittest.mock import MagicMock

import numpy as np
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
        assert 'Example data file from refnx docs' in html
        assert 'No. of data points' in html
        assert '408' in html
        assert 'Resolution function' in html
        assert 'PercentageFwhm' in html

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
        # The (previously misspelt) constraints token must be fully substituted.
        assert 'num_constriants' not in html
        assert 'num_constraints' not in html

    @staticmethod
    def _populate_fit_results(project: Project, chi2: float, n_points: int, n_pars: int) -> None:
        """Emulate a completed fit by storing results on the project's fitter."""
        fit_result = MagicMock()
        fit_result.chi2 = chi2
        fit_result.x = np.arange(n_points)
        fit_result.n_pars = n_pars
        project.fitter._fit_results = [fit_result]

    def test_compute_goodness_of_fit_na_before_fit(self, project: Project) -> None:
        # When
        summary = Summary(project)

        # Then Expect: no fit has been run, so goodness-of-fit is unavailable.
        assert summary._compute_goodness_of_fit() == 'N/A'

    def test_compute_goodness_of_fit_after_fit(self, project: Project) -> None:
        # When: reduced chi² = 20 / (14 - 4) = 2.0
        summary = Summary(project)
        self._populate_fit_results(project, chi2=20.0, n_points=14, n_pars=4)

        # Then
        gof = summary._compute_goodness_of_fit()

        # Expect
        assert gof != 'N/A'
        assert float(gof) == pytest.approx(2.0)

    def test_refinement_section_shows_goodness_of_fit(self, project: Project) -> None:
        # When
        summary = Summary(project)
        self._populate_fit_results(project, chi2=20.0, n_points=14, n_pars=4)
        gof = summary._compute_goodness_of_fit()

        # Then
        html = summary._refinement_section()

        # Expect: the template token is replaced with the actual value, not 'N/A'.
        assert gof != 'N/A'
        assert gof in html
        assert 'goodness_of_fit' not in html

    def test_save_sld_plot(self, project: Project, tmp_path) -> None:
        # When
        summary = Summary(project)
        file_path = tmp_path / 'filename'
        file_path = file_path.with_suffix('.jpg')

        # Then
        summary.save_sld_plot(file_path)

        # Expect
        assert os.path.exists(file_path)

    @pytest.mark.skip(reason='Matplotlib issue with headless CI environments')
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

    def test_figures_section_static(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary.save_sld_plot = MagicMock()
        summary.save_fit_experiment_plot = MagicMock()

        # Then
        html = summary._figures_section(interactive=False)

        # Expect
        summary.save_sld_plot.assert_called_once()
        summary.save_fit_experiment_plot.assert_called_once()
        assert 'sld_plot' in html
        assert 'fit_experiment_plot' in html

    def test_figures_section_interactive(self, project: Project) -> None:
        # When
        summary = Summary(project)
        summary.save_sld_plot = MagicMock()
        summary.save_fit_experiment_plot = MagicMock()

        # Then
        html = summary._figures_section(interactive=True)

        # Expect
        # Interactive figures must not fall back to the static image plots.
        summary.save_sld_plot.assert_not_called()
        summary.save_fit_experiment_plot.assert_not_called()
        # Two interactive plotly charts with the library embedded inline once.
        assert html.count('class="plotly-graph-div"') == 2
        assert 'Plotly.newPlot' in html
