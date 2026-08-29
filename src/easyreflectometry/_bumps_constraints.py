# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Back-port of the BUMPS inequality-constraints hook for cores that lack it.

EasyScience gained a ``constraints_factory`` argument on ``Bumps.fit`` /
``DreamSampler.sample`` / ``Sampler`` that attaches inequality penalties
to the ``FitProblem``. Cores without it accept the keyword through
``**kwargs`` and silently drop it, so the penalties are never enforced.

When the installed core is native (:data:`NATIVE`) nothing here is
installed and the library keeps passing the keyword. Otherwise
:func:`install` wraps ``build_curve_problem`` and :func:`applied`
supplies the factory for the duration of a fit, attaching the
constraints to the freshly built problem.

``FitProblem.constraints`` is a plain list read live by
``constraints_nllf()``, and ``build_curve_problem`` returns the ``Curve``
whose ``.pars`` is the very ``{prefixed name: BumpsParameter}`` mapping
the factory expects — so attaching after construction is equivalent to
passing ``constraints=`` to the constructor.

Only problems built by ``build_curve_problem`` are covered. Calling
``Bumps.fit`` with a caller-supplied ``model=`` bypasses it (that branch
constructs ``FitProblem(model)`` itself) and the constraints would be
dropped; a native core raises for that combination instead. The library
never passes ``model=``, so this is reachable only by driving the core
minimizer directly.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
from typing import Callable
from typing import Iterator
from typing import Optional

from easyscience.fitting.minimizers import minimizer_bumps
from easyscience.fitting.samplers import sampler_dream

#: Message the native core raises for non-BUMPS engines. Reproduced verbatim:
#: it is both matched by the test suite and printed in the constraints tutorial.
NON_BUMPS_ERROR = (
    "Inequality constraints (constraints_factory) require the BUMPS engine; the selected minimizer uses '{package}'."
)

#: True when the core takes ``constraints_factory`` itself, in which case the
#: library threads the keyword through and this module stays inert.
NATIVE = 'constraints_factory' in inspect.signature(minimizer_bumps.Bumps.fit).parameters

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
            # Only for parity with the native core, which warns about an
            # infeasible start point from ``FitProblem.__init__``. The
            # penalties themselves are already live without this.
            problem.model_reset()
        return problem, fit_function, curve

    build_curve_problem._easyreflectometry_shim = True
    module.build_curve_problem = build_curve_problem


def install() -> None:
    """Patch the fitting and sampling entry points, unless the core is native.

    Both consumers bind ``build_curve_problem`` at import time, so each
    namespace has to be patched; patching the defining module alone has no
    effect. Idempotent.
    """
    if NATIVE:
        return
    _patch(minimizer_bumps)
    _patch(sampler_dream)


@contextlib.contextmanager
def applied(factory: Optional[Callable]) -> Iterator[None]:
    """Attach `factory`'s constraints to problems built inside the block.

    A no-op when `factory` is ``None`` or the core is native (the keyword is
    threaded through instead).

    Parameters
    ----------
    factory : Optional[Callable]
        Receives the ``{prefixed name: BumpsParameter}`` mapping of a freshly
        built problem and returns the constraints to attach.
    """
    if factory is None or NATIVE:
        yield
        return
    install()
    token = _active.set(factory)
    try:
        yield
    finally:
        _active.reset(token)
