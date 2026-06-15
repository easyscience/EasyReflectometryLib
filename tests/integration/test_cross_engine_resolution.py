# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-engine regression tests for resolution function width conventions.

These tests verify that the same model + resolution function produces
consistent results across the refnx and refl1d engines, confirming that the
sigma/FWHM convention conversion is applied correctly at each engine
boundary.

.. note::

   The engines use different internal resolution algorithms (refnx uses
   pointwise convolution, refl1d uses oversampling), so results can differ
   by ~10-20% even with identical width conventions.  The tolerance is set
   wide enough to accommodate this while still catching a 2.355x convention
   error, which would produce >100% differences.

See GitHub issue #367 for background.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from easyreflectometry.calculators.refl1d.wrapper import Refl1dWrapper
from easyreflectometry.calculators.refnx.wrapper import RefnxWrapper
from easyreflectometry.model.resolution_functions import SIGMA_TO_FWHM
from easyreflectometry.model.resolution_functions import LinearSpline
from easyreflectometry.model.resolution_functions import PercentageFwhm
from easyreflectometry.model.resolution_functions import Pointwise


def _build_simple_model_refnx(wrapper):
    """Build a simple single-film model on a refnx wrapper."""
    wrapper.reset_storage()
    wrapper.create_material('Substrate')
    wrapper.update_material('Substrate', real=2.07, imag=0.0)
    wrapper.create_material('Film')
    wrapper.update_material('Film', real=3.45, imag=0.0)
    wrapper.create_layer('SubstrateLayer')
    wrapper.assign_material_to_layer('Substrate', 'SubstrateLayer')
    wrapper.create_layer('FilmLayer')
    wrapper.assign_material_to_layer('Film', 'FilmLayer')
    wrapper.update_layer('FilmLayer', thick=100.0, rough=3.0)
    wrapper.create_item('Item')
    wrapper.add_layer_to_item('FilmLayer', 'Item')
    wrapper.add_layer_to_item('SubstrateLayer', 'Item')
    wrapper.create_model('MyModel')
    wrapper.add_item('Item', 'MyModel')
    wrapper.update_model('MyModel', bkg=0.0)


def _build_simple_model_refl1d(wrapper):
    """Build the same single-film model on a refl1d wrapper."""
    wrapper.reset_storage()
    wrapper.create_material('Substrate')
    wrapper.update_material('Substrate', rho=2.07, irho=0.0)
    wrapper.create_material('Film')
    wrapper.update_material('Film', rho=3.45, irho=0.0)
    wrapper.create_layer('SubstrateLayer')
    wrapper.assign_material_to_layer('Substrate', 'SubstrateLayer')
    wrapper.create_layer('FilmLayer')
    wrapper.assign_material_to_layer('Film', 'FilmLayer')
    wrapper.update_layer('FilmLayer', thickness=100.0, interface=3.0)
    wrapper.create_item('Item')
    wrapper.add_layer_to_item('FilmLayer', 'Item')
    wrapper.add_layer_to_item('SubstrateLayer', 'Item')
    wrapper.create_model('MyModel')
    wrapper.add_item('Item', 'MyModel')
    wrapper.update_model('MyModel', bkg=0.0)


@pytest.mark.parametrize('resolution_pct', [1.0, 5.0, 10.0])
def test_percentage_fwhm_consistent_across_engines(resolution_pct):
    """PercentageFwhm resolution gives consistent results across engines."""
    q = np.geomspace(0.005, 0.3, 100)

    refnx_w = RefnxWrapper()
    _build_simple_model_refnx(refnx_w)
    refnx_w.set_resolution_function(PercentageFwhm(resolution_pct))
    refnx_r = refnx_w.calculate(q, 'MyModel')

    refl1d_w = Refl1dWrapper()
    _build_simple_model_refl1d(refl1d_w)
    refl1d_w.set_resolution_function(PercentageFwhm(resolution_pct))
    refl1d_r = refl1d_w.calculate(q, 'MyModel')

    assert_allclose(refnx_r, refl1d_r, rtol=0.25)


def test_linear_spline_consistent_across_engines():
    """LinearSpline resolution gives consistent results across engines.

    This is the key regression test for issue #367 -- before the fix, refl1d
    over-smeared by ~2.355x with LinearSpline because the FWHM->sigma
    conversion was missing, while refnx treated the same vector as FWHM.
    """
    q = np.geomspace(0.005, 0.3, 100)

    # A plausible per-point FWHM resolution that increases with q.
    q_knots = np.linspace(0.001, 0.5, 10)
    fwhm_knots = 0.02 * q_knots + 0.001

    refnx_w = RefnxWrapper()
    _build_simple_model_refnx(refnx_w)
    refnx_w.set_resolution_function(LinearSpline(q_knots, fwhm_knots))
    refnx_r = refnx_w.calculate(q, 'MyModel')

    refl1d_w = Refl1dWrapper()
    _build_simple_model_refl1d(refl1d_w)
    refl1d_w.set_resolution_function(LinearSpline(q_knots, fwhm_knots))
    refl1d_r = refl1d_w.calculate(q, 'MyModel')

    assert_allclose(refnx_r, refl1d_r, rtol=0.25)


def test_pointwise_consistent_across_engines():
    """Pointwise resolution (sigma from sQz) is consistent across engines.

    Before the fix, refnx treated the sigma values as FWHM and so
    under-smeared by ~2.355x relative to refl1d.
    """
    q = np.geomspace(0.005, 0.3, 100)

    # sQz is the variance of Qz; sigma = sqrt(sQz) increases with q.  The
    # magnitude mirrors the LinearSpline test (whose FWHM knots become these
    # sigma values) so both engines apply the same modest smearing.
    qz = np.linspace(0.001, 0.5, 50)
    r = np.ones_like(qz)  # only kept for serialization round-trips
    sigma = (0.02 * qz + 0.001) / SIGMA_TO_FWHM
    sqz = sigma**2

    refnx_w = RefnxWrapper()
    _build_simple_model_refnx(refnx_w)
    refnx_w.set_resolution_function(Pointwise([qz, r, sqz]))
    refnx_r = refnx_w.calculate(q, 'MyModel')

    refl1d_w = Refl1dWrapper()
    _build_simple_model_refl1d(refl1d_w)
    refl1d_w.set_resolution_function(Pointwise([qz, r, sqz]))
    refl1d_r = refl1d_w.calculate(q, 'MyModel')

    assert_allclose(refnx_r, refl1d_r, rtol=0.25)
