# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Inequality-constraint enforcement for the BUMPS engine.

EasyScience builds the ``FitProblem`` internally and offers no hook for
attaching inequality penalties to it, so the enforcement lives here
rather than in the core: :func:`install` wraps ``build_curve_problem``
and :func:`applied` makes a factory current for the duration of a fit,
attaching the constraints to each freshly built problem.

``FitProblem.constraints`` is a plain list read live by
``constraints_nllf()``, and ``build_curve_problem`` returns the ``Curve``
whose ``.pars`` is the very ``{prefixed name: BumpsParameter}`` mapping
the factory expects — so attaching after construction is equivalent to
passing ``constraints=`` to the constructor.

Only problems built by ``build_curve_problem`` are covered. Calling
``Bumps.fit`` with a caller-supplied ``model=`` bypasses it (that branch
constructs ``FitProblem(model)`` itself) and the constraints would be
dropped. The library never passes ``model=``, so this is reachable only
by driving the core minimizer directly.

Nothing here is passed to the core as a keyword: ``Bumps.fit`` accepts
arbitrary ``**kwargs`` and would swallow an unrecognised one without
complaint, which is exactly the silent-unconstrained-fit failure this
module exists to prevent.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
from typing import Callable
from typing import Iterator
from typing import Optional

from easyscience.fitting.minimizers import minimizer_bumps
from easyscience.fitting.samplers import sampler_dream

#: Raised for engines that cannot enforce inequality constraints. The wording is
#: matched by the test suite and printed in the constraints tutorial, so keep it
#: stable; ``constraints_factory`` names this library's own keyword.
NON_BUMPS_ERROR = (
    "Inequality constraints (constraints_factory) require the BUMPS engine; the selected minimizer uses '{package}'."
)

_active: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar(
    'easyreflectometry_constraints_factory', default=None
)


def _patch(module) -> None:
    """Wrap ``build_curve_problem`` in one consumer namespace."""
    original = module.build_curve_problem
    if getattr(original, '_easyreflectometry_shim', False):
        return

    @functools.wraps(original)
    def build_curve_problem(*args, **kwargs):
        problem, fit_function, curve = original(*args, **kwargs)
        factory = _active.get()
        if factory is not None:
            # ``curve.pars`` is empty when no parameter is free; the factory
            # then yields constant-only penalties, which is harmless.
            problem.constraints = list(factory(dict(curve.pars)))
            # Only to get the warning BUMPS emits for an infeasible start
            # point; the penalties themselves are already live without this.
            problem.model_reset()
        return problem, fit_function, curve

    build_curve_problem._easyreflectometry_shim = True
    module.build_curve_problem = build_curve_problem


def install() -> None:
    """Patch the fitting and sampling entry points.

    Both consumers bind ``build_curve_problem`` at import time, so each
    namespace has to be patched; patching the defining module alone has no
    effect. Idempotent.
    """
    _patch(minimizer_bumps)
    _patch(sampler_dream)


def is_applied() -> bool:
    """Whether an :func:`applied` block is active in the current context.

    Lets an inner wrapper (``MultiFitter`` routes the raw
    ``easy_science_multi_fitter.fit`` through the constraints machinery)
    detect that an outer block already attached a factory — possibly an
    explicit one that must not be overridden by re-resolving the provider.
    """
    return _active.get() is not None


@contextlib.contextmanager
def applied(factory: Optional[Callable]) -> Iterator[None]:
    """Attach `factory`'s constraints to problems built inside the block.

    A no-op when `factory` is ``None``.

    Parameters
    ----------
    factory : Optional[Callable]
        Receives the ``{prefixed name: BumpsParameter}`` mapping of a freshly
        built problem and returns the constraints to attach.
    """
    if factory is None:
        yield
        return
    install()
    token = _active.set(factory)
    try:
        yield
    finally:
        _active.reset(token)
