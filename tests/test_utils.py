# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyreflectometry import Project
from easyreflectometry.utils import count_fixed_parameters
from easyreflectometry.utils import count_free_parameters


def test_count_free_parameters():
    # When
    project = Project()
    project.default_model()
    project.parameters[0].free = True

    # Then
    count = count_free_parameters(project)

    # Expect
    assert count == 1


def test_count_fixed_parameters():
    # When
    project = Project()
    project.default_model()
    project.parameters[0].free = True

    # Then
    count = count_fixed_parameters(project)

    # Expect
    assert count == 13
