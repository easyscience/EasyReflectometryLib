Fitting
=======

.. currentmodule:: easyreflectometry.fitting

Objective functions and non-positive variance handling
-----------------------------------------------------

:class:`MultiFitter` supports several objective modes for handling reflectometry
data during fitting, especially when measured variances are non-positive.

The default objective is ``hybrid``. This uses ordinary weighted least squares
for points with positive variance and applies a Mighell-style substitution only
to points whose variance is non-positive. The older ``legacy_mask`` mode
drops non-positive-variance points before fitting. The ``mighell`` mode applies the
Mighell transform to every point.

Mighell objective
~~~~~~~~~~~~~~~~~

The full ``mighell`` objective follows the algebraic form of the
``chi^2_gamma`` statistic described by Mighell for Poisson-distributed count
data:

.. math::

   \chi^2_\gamma =
   \sum_i \frac{[n_i + \min(n_i, 1) - m_i]^2}{n_i + 1}

where ``n_i`` are observed counts and ``m_i`` are model values.

In EasyReflectometry this is implemented as a weighted least-squares problem.
For each observed value ``y_i`` the fitted target is shifted to

.. math::

   y_{\mathrm{eff},i} = y_i + \min(y_i, 1)

and the effective uncertainty is

.. math::

   \sigma_i = \sqrt{y_i + 1}

so the minimized objective is

.. math::

   \sum_i \left(\frac{y_{\mathrm{eff},i} - f_i}{\sigma_i}\right)^2 =
   \sum_i \frac{[y_i + \min(y_i, 1) - f_i]^2}{y_i + 1}

This is the same algebraic form as Mighell's statistic, with the model value
``f_i`` replacing ``m_i``.

Scope and interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

Mighell's statistic was derived for Poisson-distributed count data. In
reflectometry workflows, the fitted values are usually normalized
reflectivities or intensities rather than raw counts. They may already have
been processed, scaled, background-corrected, or otherwise transformed before
they reach the fitter.

This distinction matters when interpreting the result. The full ``mighell``
objective is not only a reweighting of residuals; it also changes the fitted
target from ``y`` to ``y + min(y, 1)``. For values between zero and one, this
can substantially increase the target value. A fit can therefore have a good
Mighell objective value while looking poorer against the originally plotted
reflectivity curve, or while having a worse classical chi-square.

For reflectometry data, ``hybrid`` is generally the recommended compromise:
it preserves ordinary weighted least-squares behavior where positive variances are
available, while still allowing non-positive-variance points to contribute through the
Mighell-style substitution.

Objective modes
~~~~~~~~~~~~~~~

``hybrid``
   Default. Use standard weighted least squares for points with positive
   variance and apply the Mighell substitution only where variance is
   non-positive.

``mighell``
   Apply the Mighell transform to all points. The reported objective chi-square
   is evaluated in transformed objective space and should not be interpreted as
   a classical chi-square against the original reflectivity values.

``legacy_mask``
   Remove non-positive-variance points before fitting and use standard weighted least
   squares for the remaining points.

``auto``
   Alias for ``hybrid``.

Fit metrics
~~~~~~~~~~~

The fitter exposes both objective-space and classical fit metrics after fitting.
``objective_chi2`` and ``objective_reduced_chi`` describe the minimized
objective, which may include transformed targets under ``hybrid`` or
``mighell``. ``classical_chi2`` and ``classical_reduced_chi`` are computed
against the original observed reflectivity values using only points with
positive variance.

API reference
-------------

.. automodule:: easyreflectometry.fitting
   :members:
   :undoc-members:
   :show-inheritance: