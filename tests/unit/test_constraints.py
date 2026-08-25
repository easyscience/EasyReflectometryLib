# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the user-facing constraint helpers
"""

import pytest
from easyscience import global_object
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

from easyreflectometry import constrain
from easyreflectometry import constrain_equal
from easyreflectometry import unconstrain
from easyreflectometry.project import Project


@pytest.fixture(autouse=True)
def clear_global_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


class TestConstrainEqual:
    def test_ties_value_and_follows(self):
        leader = Parameter('leader', 5.0, unit='angstrom', min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, unit='angstrom', min=0.0, max=2.0)

        constrain_equal(follower, to=leader)

        assert follower.independent is False
        assert follower.value == 5.0
        leader.value = 7.0
        assert follower.value == 7.0

    def test_overwrites_bounds_and_clears_fixed(self):
        leader = Parameter('leader', 5.0, unit='angstrom', min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, unit='angstrom', min=0.5, max=2.0, fixed=True)

        constrain_equal(follower, to=leader)

        assert follower.min == leader.min
        assert follower.max == leader.max
        assert follower.fixed is False

    def test_dependent_setters_are_locked(self):
        leader = Parameter('leader', 5.0)
        follower = Parameter('follower', 1.0)
        constrain_equal(follower, to=leader)

        with pytest.raises(AttributeError):
            follower.value = 3.0
        with pytest.raises(AttributeError):
            follower.fixed = True

    def test_descriptor_dependency_gives_degenerate_bounds(self):
        # A DescriptorNumber dependency evaluates to a non-Parameter, so
        # EasyScience sets min == max == value on the dependent. Pinned
        # here because it is why the docs say "reset bounds after
        # unconstrain".
        leader = DescriptorNumber('leader', 3.0)
        follower = Parameter('follower', 1.0, min=0.0, max=2.0)

        constrain_equal(follower, to=leader)
        assert follower.min == 3.0
        assert follower.max == 3.0

        unconstrain(follower)
        assert follower.min == 3.0
        assert follower.max == 3.0


class TestConstrain:
    def test_scale_expression_follows(self):
        leader = Parameter('leader', 5.0, unit='angstrom', min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, unit='angstrom', min=0.0, max=2.0)

        constrain(follower, '2 * t', t=leader)

        assert follower.value == 10.0
        leader.value = 6.0
        assert follower.value == 12.0

    def test_multi_parameter_expression(self):
        fraction = Parameter('fraction', 0.25, min=0.0, max=1.0)
        sld_a = Parameter('sld_a', 2.0)
        sld_b = Parameter('sld_b', 6.0)
        mixed = Parameter('mixed', 0.0)

        constrain(mixed, 'frac * a + (1 - frac) * b', frac=fraction, a=sld_a, b=sld_b)

        assert mixed.value == pytest.approx(0.25 * 2.0 + 0.75 * 6.0)
        fraction.value = 0.5
        assert mixed.value == pytest.approx(0.5 * 2.0 + 0.5 * 6.0)

    def test_reconstrain_replaces_dependency(self):
        first = Parameter('first', 1.0)
        second = Parameter('second', 2.0)
        follower = Parameter('follower', 0.0)

        constrain_equal(follower, to=first)
        constrain_equal(follower, to=second)
        assert follower.value == 2.0

        # No stale updates from the previous target
        first.value = 100.0
        assert follower.value == 2.0
        second.value = 3.0
        assert follower.value == 3.0

    def test_unknown_name_raises_and_reverts(self):
        leader = Parameter('leader', 5.0)
        follower = Parameter('follower', 1.0)

        with pytest.raises(NameError):
            constrain(follower, 'a + b', a=leader)

        assert follower.independent is True
        assert follower.value == 1.0


class TestUnconstrain:
    def test_removes_constraint_and_keeps_last_value(self):
        leader = Parameter('leader', 5.0, min=0.0, max=10.0)
        follower = Parameter('follower', 1.0, min=0.0, max=2.0)
        constrain_equal(follower, to=leader)
        leader.value = 7.0

        unconstrain(follower)

        assert follower.independent is True
        assert follower.value == 7.0
        # Bounds and fixed state are not restored
        assert follower.min == 0.0
        assert follower.max == 10.0
        assert follower.fixed is False
        # Fittable again
        follower.value = 1.5
        assert follower.value == 1.5
        assert leader.value == 7.0

    def test_idempotent_on_independent_parameter(self):
        parameter = Parameter('parameter', 1.0)
        unconstrain(parameter)
        unconstrain(parameter)
        assert parameter.independent is True


class TestProjectRoundTrip:
    def test_constraint_survives_as_dict_from_dict(self):
        # Requires the easyscience serializer to route nested Parameters through
        # ``Parameter.as_dict`` and park the dependency as pending on rebuild.
        src_project = Project()
        src_project._info['name'] = 'Test'
        src_project.default_model()
        src_project._with_experiments = False
        sample = src_project.models[0].sample
        leader = sample[1].layers[0].thickness
        follower = sample[2].layers[0].thickness
        constrain(follower, '2 * t', t=leader)
        assert follower.value == 2 * leader.value

        project_dict = src_project.as_dict()
        global_object.map._clear()

        project = Project()
        project.from_dict(project_dict)
        sample = project.models[0].sample
        leader = sample[1].layers[0].thickness
        follower = sample[2].layers[0].thickness

        assert follower.independent is False
        leader.value = 60.0  # within the default [50, 200] thickness limits
        assert follower.value == 120.0
