# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for CalculatorFactory serialization."""

import pickle  # noqa: S403

import numpy as np
from numpy.testing import assert_allclose

from easyreflectometry.calculators import CalculatorFactory
from easyreflectometry.model import Model
from easyreflectometry.model import PercentageFwhm
from easyreflectometry.sample import Layer
from easyreflectometry.sample import Material
from easyreflectometry.sample import Multilayer
from easyreflectometry.sample import Sample


def test_calculator_factory_pickle_preserves_active_wrapper_storage():
    """Pickled calculator factories retain model storage for worker processes."""
    si = Material(sld=2.07, isld=0.0, name='Si')
    film = Material(sld=2.0, isld=0.0, name='Film')
    d2o = Material(sld=6.36, isld=0.0, name='D2O')

    sample = Sample(
        Multilayer(Layer(material=si, thickness=0.0, roughness=3.0, name='Si')),
        Multilayer(Layer(material=film, thickness=250.0, roughness=3.0, name='Film')),
        Multilayer(Layer(material=d2o, thickness=0.0, roughness=3.0, name='D2O')),
    )
    model = Model(
        sample=sample,
        scale=1.0,
        background=1e-6,
        resolution_function=PercentageFwhm(0.02),
    )
    interface = CalculatorFactory()
    interface.switch('refnx')
    model.interface = interface

    restored = pickle.loads(pickle.dumps(interface))  # noqa: S301

    assert model.unique_name in restored()._wrapper.storage['model']
    q = np.linspace(0.01, 0.3, 10)
    assert_allclose(
        restored.fit_func(q, model.unique_name),
        interface.fit_func(q, model.unique_name),
    )
