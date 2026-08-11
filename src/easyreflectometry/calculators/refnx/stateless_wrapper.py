# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter
from refnx import analysis
from refnx import reflect

from easyreflectometry.model import PercentageFwhm
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM

from ..stateless_wrapper_base import StatelessWrapperBase

if TYPE_CHECKING:
    from easyreflectometry.model import Model
    from easyreflectometry.sample import BaseAssembly
    from easyreflectometry.sample import Layer


class WriteBackParameter(analysis.Parameter):
    """A refnx parameter which forwards every assignment to easyscience.

    The proxy is what makes the *push* half of the backend-to-core contract
    work: anything refnx assigns during an evaluation (a constraint, bounds
    clipping, a refnx-side fitter) lands in the core ``Parameter`` immediately
    and fires its observers.

    It reads the core value once, at construction, which is the whole "read the
    model at evaluation time" story: the proxy lives for the duration of a
    single calculation and is garbage together with the structure it was built
    into.
    """

    def __init__(self, easyscience_parameter: Parameter):
        """Constructor.

        Parameters
        ----------
        easyscience_parameter : Parameter
            The core parameter this proxy stands in for.
        """
        # Set before super().__init__ so the overridden setter below can tell
        # a constructor-time write from a backend write.
        self._easyscience_parameter = None
        super().__init__(easyscience_parameter.value, name=easyscience_parameter.unique_name)
        self._easyscience_parameter = easyscience_parameter

    @property
    def easyscience_parameter(self) -> Parameter:
        """The core parameter this proxy writes back to."""
        return self._easyscience_parameter

    @analysis.Parameter.value.setter
    def value(self, value: float) -> None:
        """Set the value on the refnx side and push it to easyscience.

        Parameters
        ----------
        value : float
            The new value.
        """
        analysis.Parameter.value.fset(self, value)
        if self._easyscience_parameter is None:
            return
        # Forward the POST-set value rather than the raw input: should refnx's
        # own setter clamp or transform, core must receive what the backend
        # actually holds instead of disagreeing with it.
        self._easyscience_parameter.update_from_calculator(float(analysis.Parameter.value.fget(self)))

    def sync(self) -> bool:
        """Push the current backend value to easyscience if it has drifted.

        The setter above covers everything refnx *assigns*, but a refnx-side
        constraint is evaluated lazily on read and never goes through it. One
        comparison per proxy at the end of an evaluation catches those without
        turning reads into I/O: when nothing moved, which is the normal case in
        a fit iteration, this does nothing at all.

        Returns
        -------
        bool
            Whether a value was pushed.
        """
        if self._easyscience_parameter is None:
            return False
        backend_value = float(analysis.Parameter.value.fget(self))
        if backend_value == float(self._easyscience_parameter.value):
            return False
        self._easyscience_parameter.update_from_calculator(backend_value)
        return True


def _proxy(easyscience_parameter: Parameter, collector: list[WriteBackParameter] | None = None) -> WriteBackParameter:
    """Wrap a core parameter in a write-through refnx parameter.

    Parameters
    ----------
    easyscience_parameter : Parameter
        The core parameter to wrap.
    collector : list[WriteBackParameter] | None, optional
        When given, the proxy is recorded so the caller can push lazily
        evaluated backend values back after the evaluation. By default, None.
    """
    proxy = WriteBackParameter(easyscience_parameter)
    if collector is not None:
        collector.append(proxy)
    return proxy


def _parameter(obj: object, name: str) -> Parameter:
    """Return the ``Parameter`` backing ``obj.<name>``.

    ``MaterialMixture`` exposes its derived ``sld`` / ``isld`` as plain floats
    (they are computed from the constituent fractions), keeping the
    ``Parameter`` objects on the private attributes. Everything else exposes
    the ``Parameter`` directly.
    """
    value = getattr(obj, name)
    if isinstance(value, Parameter):
        return value
    return getattr(obj, f'_{name}')


class RefnxStatelessWrapper(StatelessWrapperBase):
    """Build refnx objects from the easyscience model, per evaluation.

    There is no ``storage``: every ``Structure`` / ``Slab`` / ``SLD`` is
    constructed from the current model when a calculation is requested and
    dropped when it returns. Structural edits therefore need no mirroring, the
    next evaluation simply sees the new object graph.
    """

    @property
    def include_magnetism(self) -> bool:
        """Include magnetism."""
        return self._magnetism

    @include_magnetism.setter
    def include_magnetism(self, magnetism: bool) -> None:
        """Set the magnetism flag.

        Parameters
        ----------
        magnetism : bool
            The magnetism flag.
        """
        raise NotImplementedError('Magnetism is not supported by refnx')

    # ----- calculation -----

    def calculate(self, q_array: np.ndarray, model_id: str | None = None) -> np.ndarray:
        """For a given q array calculate the corresponding reflectivity.

        Parameters
        ----------
        q_array : np.ndarray
            Array of data points to be calculated.
        model_id : str | None, optional
            The ``unique_name`` of the model to evaluate. May be omitted when
            exactly one model is attached. By default, None.

        Returns
        -------
        np.ndarray
            Reflectivity calculated at q.
        """
        model = self._get_model(model_id)
        proxies: list[WriteBackParameter] = []
        structure = self._build_structure(model, proxies)
        refl_model = reflect.ReflectModel(
            structure,
            scale=_proxy(model.scale, proxies),
            bkg=_proxy(model.background, proxies),
            dq_type='pointwise',
        )
        reflectivity = refl_model(x=q_array, x_err=self._resolution_vector(q_array))
        for proxy in proxies:
            proxy.sync()
        return reflectivity

    def sld_profile(self, model_id: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the scattering length density profile.

        Parameters
        ----------
        model_id : str | None, optional
            The ``unique_name`` of the model to evaluate. May be omitted when
            exactly one model is attached. By default, None.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            z and sld(z).
        """
        return self._build_structure(self._get_model(model_id)).sld_profile()

    def _resolution_vector(self, q_array: np.ndarray) -> np.ndarray | float:
        """Return the resolution width in the convention refnx expects.

        refnx reads a **scalar** ``x_err`` as a constant dq/q FWHM percentage,
        and a per-point ``x_err`` as the FWHM at each q. ``smearing`` returns
        sigma, hence the conversion.
        """
        if isinstance(self._resolution_function, PercentageFwhm):
            return self._resolution_function.constant
        return self._resolution_function.smearing(q_array) * SIGMA_TO_FWHM

    # ----- model traversal -----

    def _build_structure(self, model: Model, proxies: list[WriteBackParameter] | None = None) -> reflect.Structure:
        """Build a refnx structure from the easyscience model.

        The sample is walked in order: assemblies in insertion order (which is
        the physical order) and, within each assembly, layers from fronting to
        substrate. An assembly which repeats once is flattened into its slabs,
        matching what the legacy storage-based build produced.

        Parameters
        ----------
        model : Model
            The model to walk.
        proxies : list[WriteBackParameter] | None, optional
            Collector for the write-through proxies created along the way. By
            default, None.

        Returns
        -------
        reflect.Structure
            The structure to calculate from.
        """
        components = []
        for assembly in model.sample:
            slabs = [self._slab(layer, proxies) for layer in assembly.layers]
            repetitions = self._repetitions(assembly)
            if repetitions is None or repetitions.value == 1:
                components.extend(slabs)
            else:
                stack = reflect.Stack(name=assembly.unique_name)
                for slab in slabs:
                    stack.append(slab)
                stack.repeats = _proxy(repetitions, proxies)
                components.append(stack)
        structure = reflect.Structure(components)
        self._log_structure(model, structure)
        return structure

    def _slab(self, layer: Layer, proxies: list[WriteBackParameter] | None = None) -> reflect.Slab:
        """Build a refnx slab from an easyscience layer."""
        return reflect.Slab(
            _proxy(_parameter(layer, 'thickness'), proxies),
            self._sld(layer.material, proxies),
            _proxy(_parameter(layer, 'roughness'), proxies),
            name=layer.unique_name,
        )

    @staticmethod
    def _sld(material: object, proxies: list[WriteBackParameter] | None = None) -> reflect.SLD:
        """Build a refnx scatterer from an easyscience material.

        ``MaterialMixture`` is handled identically to ``Material``: its derived
        ``sld`` / ``isld`` feed the same ``SLD.real`` / ``SLD.imag``.
        """
        sld = reflect.SLD(0, name=material.unique_name)
        sld.real = _proxy(_parameter(material, 'sld'), proxies)
        sld.imag = _proxy(_parameter(material, 'isld'), proxies)
        return sld

    @staticmethod
    def _repetitions(assembly: BaseAssembly) -> Parameter | None:
        """Return the repetitions parameter of an assembly, if it has one."""
        repetitions = getattr(assembly, 'repetitions', None)
        if repetitions is None:
            return None
        return repetitions if isinstance(repetitions, Parameter) else assembly._repetitions

    @staticmethod
    def _log_structure(model: Model, structure: reflect.Structure) -> None:
        """Log the built structure at debug level.

        The legacy ``wrapper.storage`` doubled as an inspection point for what
        the backend was actually given. Since nothing is stored any more, the
        constructed structure is logged instead so it stays observable, one
        record per evaluation.
        """
        logger = global_object.log.getLogger('easyreflectometry.calculators.refnx')
        if not logger.isEnabledFor(10):  # logging.DEBUG
            return
        logger.debug(
            'Built refnx structure for model %s: %s',
            model.unique_name,
            [
                {
                    'name': component.name,
                    'repeats': getattr(getattr(component, 'repeats', None), 'value', None),
                    'thick': getattr(getattr(component, 'thick', None), 'value', None),
                    'rough': getattr(getattr(component, 'rough', None), 'value', None),
                    'sld': getattr(getattr(getattr(component, 'sld', None), 'real', None), 'value', None),
                }
                for component in structure.components
            ],
        )
