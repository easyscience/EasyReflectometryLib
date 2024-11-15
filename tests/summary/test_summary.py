import os

import pytest
from easyscience import global_object

import easyreflectometry
from easyreflectometry import Project
from easyreflectometry.summary import Summary
from easyreflectometry.summary.html_templates import HTML_TEMPLATE

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

    # def test_compile_html_summary(self, project: Project) -> None:
    #     # When
    #     summary = Summary(project)

    #     # Then
    #     result = summary.compile_html_summary()

    #     # Expect
    #     assert result is not None

    def test_set_project_information_section(self, project: Project) -> None:
        # When
        project._created = True
        summary = Summary(project)
        html = 'project_information_section'

        # Then
        html = summary._set_project_information_section(html)

        # Expect
        assert (
            html
            == '\n<tr>\n    <td><h3>Project information</h3></td>\n</tr>\n\n<tr></tr>\n\n<tr>\n    <th>Title</th>\n    <th>ExampleProject</th>\n</tr>\n<tr>\n    <td>Description</td>\n    <td>Reflectometry, 1D</td>\n</tr>\n<tr>\n    <td>No. of experiments</td>\n    <td>0</td>\n</tr>\n\n<tr></tr>\n'
        )

    def test_set_experiments_section(self, project: Project) -> None:
        # When
        project._created = True
        fpath = os.path.join(PATH_STATIC, 'example.ort')
        project.load_experiment_for_model_at_index(fpath)
        summary = Summary(project)
        html = 'experiment_section'

        # Then
        html = summary._set_experiments_section(html)

        # Expect
        assert html == ''

    def test_set_refinement_section(self, project: Project) -> None:
        # When
        project._created = True
        summary = Summary(project)
        html = 'refinement_section'

        # Then
        html = summary._set_refinement_section(html)

        # Expect
        assert (
            html
            == '\n<tr>\n    <td><h3>Refinement</h3></td>\n</tr>\n\n<tr></tr>\n\n<tr>\n    <td>Calculation engine</td>\n    <td>refnx</td>\n</tr>\n<tr>\n    <td>Minimization engine</td>\n    <td>LMFit_leastsq</td>\n</tr>\n<tr>\n    <td>Goodness-of-fit: reduced <i>&chi;</i><sup>2</sup></td>\n    <td>goodness_of_fit</td>\n</tr>\n<tr>\n    <td>No. of parameters: total, free, fixed</td>\n    <td>14, 0, 14</td>\n</tr>\n<tr>\n    <td>No. of constraints</td>\n    <td>num_constriants</td>\n</tr>\n\n<tr></tr>\n'
        )
