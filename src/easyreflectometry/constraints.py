# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""User-facing helpers for constraining model parameters.

These functions are thin wrappers around the EasyScience
parameter-dependency mechanism (``Parameter.make_dependent_on`` /
``Parameter.make_independent``). The dependency graph is owned entirely
by EasyScience; constrained parameters are excluded from the free fit
parameters.

Note: custom constraints are not yet preserved when a project is saved
and reloaded. The nested-parameter serialization in EasyScience drops
the dependency information. Re-apply constraints after loading a
project.
"""

from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter

__all__ = ['constrain', 'constrain_equal', 'unconstrain']


def constrain_equal(parameter: Parameter, to: DescriptorNumber) -> None:
    """Tie `parameter` to always equal `to`.

    `parameter` becomes dependent: it is removed from the free fit
    parameters, immediately takes the value of `to`, and follows it from
    then on. Its value, unit, variance, min and max are replaced by
    `to`'s, and its `fixed` flag is cleared. While constrained, the
    parameter's value and bounds cannot be set directly.

    Parameters
    ----------
    parameter : Parameter
        Parameter to make dependent (the follower).
    to : DescriptorNumber
        Parameter (or descriptor) to follow.
    """
    parameter.make_dependent_on(dependency_expression='a', dependency_map={'a': to})


def constrain(parameter: Parameter, expression: str, **parameters: DescriptorNumber) -> None:
    """Tie `parameter` to an arbitrary expression of other parameters.

    Placeholders in `expression` are supplied as keyword arguments:

    .. code-block:: python

        constrain(layer_b.thickness, '2 * t', t=layer_a.thickness)

    `parameter` becomes dependent; its value, unit, variance, min and max
    are replaced by the evaluated expression and its `fixed` flag is
    cleared. Placeholder names must be valid Python identifiers and not
    Python keywords. An unmapped name in the expression that matches a
    mathematical builtin (`e`, `pi`, `sin`, ...) evaluates silently
    instead of raising `NameError`, so prefer descriptive placeholder
    names.

    Parameters
    ----------
    parameter : Parameter
        Parameter to make dependent (the follower).
    expression : str
        Mathematical expression to evaluate, e.g. ``'2 * t'``.
    parameters : DescriptorNumber
        Placeholder-name to parameter mapping for `expression`.
    """
    parameter.make_dependent_on(dependency_expression=expression, dependency_map=parameters)


def unconstrain(parameter: Parameter) -> None:
    """Remove any constraint from `parameter`.

    Idempotent: calling it on an already-independent parameter is a
    no-op. The parameter keeps its current (last evaluated) value and
    becomes fittable again. Its bounds, unit, variance and `fixed` state
    are not restored to their pre-constraint values. Review and reset
    the bounds before fitting.

    Parameters
    ----------
    parameter : Parameter
        Parameter to make independent again.
    """
    if not parameter.independent:
        parameter.make_independent()
