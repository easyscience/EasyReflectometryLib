# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from easyscience import global_object
from easyscience.variable import Parameter
from refl1d import names
from refl1d.sample.layers import Repeat

from ..stateless_wrapper_base import StatelessWrapperBase

if TYPE_CHECKING:
    from easyreflectometry.model import Model
    from easyreflectometry.sample import BaseAssembly
    from easyreflectometry.sample import Layer

RESOLUTION_PADDING = 3.5
OVERSAMPLING_FACTOR = 21
ALL_POLARIZATIONS = False


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


class Refl1dStatelessWrapper(StatelessWrapperBase):
    """Build refl1d objects from the easyscience model, per evaluation.

    There is no ``storage``: the whole ``Stack`` is constructed from the
    current model when a calculation is requested and dropped when it returns.

    Unlike the refnx wrapper, no write-through proxies are created: refl1d
    never assigns to a parameter during a reflectivity calculation (we build
    every bumps object ourselves and attach no bumps-side constraints), so
    there is nothing to push back and :meth:`reconcile` stays empty.
    """

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
        sample = self._build_sample(model)
        # smearing() returns sigma, which is exactly what refl1d's probe.dQ expects.
        dq_array = self._resolution_function.smearing(q_array)

        if not self._magnetism:
            probe = _get_probe(
                q_array=q_array,
                dq_array=dq_array,
                model=model,
                oversampling_factor=OVERSAMPLING_FACTOR,
            )
            # returns q, reflectivity
            _, reflectivity = names.Experiment(probe=probe, sample=sample).reflectivity()
        else:
            polarized_probe = _get_polarized_probe(
                q_array=q_array,
                dq_array=dq_array,
                model=model,
                oversampling_factor=OVERSAMPLING_FACTOR,
                all_polarizations=ALL_POLARIZATIONS,
            )
            polarized_reflectivity = names.Experiment(probe=polarized_probe, sample=sample).reflectivity()

            if ALL_POLARIZATIONS:
                raise NotImplementedError('Polarized reflectivity not yet implemented')
            # Only pick the pp reflectivity, returns q, reflectivity
            _, reflectivity = polarized_reflectivity[0]

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
        model = self._get_model(model_id)
        sample = self._build_sample(model)
        probe = _get_probe(
            q_array=np.array([1]),  # dummy value
            dq_array=np.array([1]),  # dummy value
            model=model,
        )
        z, sld, _ = names.Experiment(probe=probe, sample=sample).smooth_profile()
        # -1 to reverse the order
        return z, sld[::-1]

    # ----- model traversal -----

    def _build_sample(self, model: Model) -> names.Stack:
        """Build a refl1d stack from the easyscience model.

        refl1d stacks from the substrate up, the opposite of the sample order,
        hence the reversals. An assembly which repeats once is flattened into
        its slabs, matching what the legacy storage-based build produced.

        Parameters
        ----------
        model : Model
            The model to walk.

        Returns
        -------
        names.Stack
            The sample to calculate from.
        """
        sample = names.Stack()
        # -1 to reverse the order
        for assembly in list(model.sample)[::-1]:
            slabs = [self._slab(layer) for layer in assembly.layers]
            repetitions = self._repetitions(assembly)
            if repetitions is None or repetitions.value == 1:
                for slab in slabs[::-1]:
                    sample |= slab
            else:
                stack = names.Stack()
                for slab in slabs[::-1]:
                    stack |= slab
                sample |= Repeat(stack, repeat=repetitions.value)
        self._log_sample(model, sample)
        return sample

    def _slab(self, layer: Layer) -> names.Slab:
        """Build a refl1d slab from an easyscience layer."""
        magnetism = names.Magnetism(rhoM=0.0, thetaM=0.0) if self._magnetism else None
        return names.Slab(
            material=self._material(layer.material),
            thickness=_parameter(layer, 'thickness').value,
            interface=_parameter(layer, 'roughness').value,
            name=str(layer.unique_name),
            magnetism=magnetism,
        )

    @staticmethod
    def _material(material: object) -> names.SLD:
        """Build a refl1d scatterer from an easyscience material.

        ``MaterialMixture`` is handled identically to ``Material``: its derived
        ``sld`` / ``isld`` feed the same ``rho`` / ``irho``.
        """
        return names.SLD(
            str(material.unique_name),
            rho=_parameter(material, 'sld').value,
            irho=_parameter(material, 'isld').value,
        )

    @staticmethod
    def _repetitions(assembly: BaseAssembly) -> Parameter | None:
        """Return the repetitions parameter of an assembly, if it has one."""
        repetitions = getattr(assembly, 'repetitions', None)
        if repetitions is None:
            return None
        return repetitions if isinstance(repetitions, Parameter) else assembly._repetitions

    @staticmethod
    def _log_sample(model: Model, sample: names.Stack) -> None:
        """Log the built sample at debug level.

        The legacy ``wrapper.storage`` doubled as an inspection point for what
        the backend was actually given. Since nothing is stored any more, the
        constructed sample is logged instead so it stays observable, one record
        per evaluation.
        """
        logger = global_object.log.getLogger('easyreflectometry.calculators.refl1d')
        if not logger.isEnabledFor(10):  # logging.DEBUG
            return
        logger.debug('Built refl1d sample for model %s: %s', model.unique_name, repr(sample))


def _get_oversampling_q(q_array: np.ndarray, dq_array: np.ndarray, oversampling_factor: int) -> np.ndarray:
    """Get oversampling q."""
    argmin = np.argmin(q_array)  # index of the smallest q element
    argmax = np.argmax(q_array)  # index of the largest q element
    return np.linspace(
        q_array[argmin] - RESOLUTION_PADDING * dq_array[argmin],  # dq at the smallest q index
        q_array[argmax] + RESOLUTION_PADDING * dq_array[argmax],  # dq at the largest q index
        oversampling_factor * len(q_array),
    )


def _get_probe(
    q_array: np.ndarray,
    dq_array: np.ndarray,
    model: Model,
    oversampling_factor: int = 1,
    magnetism: bool = False,
) -> names.QProbe:
    """Get probe."""
    probe = names.QProbe(
        Q=q_array,
        dQ=dq_array,
        intensity=model.scale.value,
        background=model.background.value,
    )

    # Add theta_offset attribute if magnetism is enabled
    # This is required for PolarizedQProbe to work correctly
    if magnetism:
        probe.theta_offset = names.Parameter.default(0, name='theta_offset')

    if oversampling_factor > 1:
        probe.calc_Qo = _get_oversampling_q(q_array, dq_array, oversampling_factor)
    return probe


def _get_polarized_probe(
    q_array: np.ndarray,
    dq_array: np.ndarray,
    model: Model,
    oversampling_factor: int = 1,
    all_polarizations: bool = False,
) -> names.PolarizedNeutronQProbe:
    """Get polarized probe."""
    four_probes = []
    for i in range(4):
        if i == 0 or all_polarizations:
            probe = _get_probe(
                q_array=q_array,
                dq_array=dq_array,
                model=model,
                oversampling_factor=oversampling_factor,
                magnetism=True,  # Enable magnetism for polarized probes
            )
        else:
            probe = None
        four_probes.append(probe)

    # Create polarized probe and work around initialization bug
    polarized_probe = names.PolarizedNeutronQProbe.__new__(names.PolarizedNeutronQProbe)
    polarized_probe._union_cache_key = None  # Initialize missing attribute
    polarized_probe.__init__(xs=four_probes, name='polarized')
    return polarized_probe
