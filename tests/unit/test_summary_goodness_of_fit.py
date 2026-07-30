# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the summary goodness-of-fit computation and refinement section."""

from unittest.mock import MagicMock

import pytest
from easyscience import global_object

from easyreflectometry import Project
from easyreflectometry.summary import Summary


@pytest.fixture
def project() -> Project:
    global_object.map._clear()
    project = Project()
    project.default_model()
    return project


def _fit_result(chi2=None, reduced_chi2=None, n_points=0, n_pars=0):
    result = MagicMock()
    result.chi2 = chi2
    result.reduced_chi2 = reduced_chi2
    result.x = list(range(n_points))
    result.n_pars = n_pars
    return result


class TestComputeGoodnessOfFit:
    def test_returns_na_when_no_fit_has_been_run(self, project: Project):
        summary = Summary(project)
        assert summary._compute_goodness_of_fit() == 'N/A'

    def test_returns_na_when_last_fit_results_is_empty(self, project: Project):
        project._last_fit_results = []
        summary = Summary(project)
        assert summary._compute_goodness_of_fit() == 'N/A'

    def test_single_result_uses_its_reduced_chi2(self, project: Project):
        project._last_fit_results = [_fit_result(reduced_chi2=1.2345)]
        summary = Summary(project)
        assert summary._compute_goodness_of_fit() == '1.234'

    def test_multiple_results_aggregate_over_global_dof(self, project: Project):
        # total chi2 = 30, total points = 16, n_pars = 6 -> dof = 10 -> gof = 3
        project._last_fit_results = [
            _fit_result(chi2=10.0, n_points=8, n_pars=6),
            _fit_result(chi2=20.0, n_points=8, n_pars=6),
        ]
        summary = Summary(project)
        assert summary._compute_goodness_of_fit() == '3'

    def test_multiple_results_with_nonpositive_dof_return_zero(self, project: Project):
        project._last_fit_results = [
            _fit_result(chi2=10.0, n_points=2, n_pars=6),
            _fit_result(chi2=20.0, n_points=2, n_pars=6),
        ]
        summary = Summary(project)
        assert summary._compute_goodness_of_fit() == '0'

    def test_returns_na_when_result_values_are_invalid(self, project: Project):
        project._last_fit_results = [_fit_result(reduced_chi2='not-a-number')]
        summary = Summary(project)
        assert summary._compute_goodness_of_fit() == 'N/A'


class TestRefinementSection:
    def test_refinement_section_renders_counts_and_gof(self, project: Project):
        project._last_fit_results = [_fit_result(reduced_chi2=2.5)]
        summary = Summary(project)

        html = summary._refinement_section()

        assert '2.5' in html
        # every placeholder must have been substituted with a number
        for placeholder in (
            'num_total_params',
            'num_free_params',
            'num_fixed_params',
            'num_constriants',
            'num_constraints',
            'goodness_of_fit',
        ):
            assert placeholder not in html
