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


def test_count_parameter_user_constraints():
    # When
    project = Project()
    project.default_model()
    # project.parameters[0].user_constraints['name_other_parameter'] = 'constraint'

    # # Then
    # count = count_parameter_user_constraints(project)

    # # Expect
    # assert count == 1
