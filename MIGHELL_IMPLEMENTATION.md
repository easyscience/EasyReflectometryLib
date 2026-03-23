# Mighell Implementation Notes

## Summary

The `mighell` objective implemented in `easyreflectometry.fitting` is mathematically consistent with the
`chi^2_gamma` statistic described in `mighell_fitting.pdf`.

However, the paper derives this statistic for Poisson-distributed count data, whereas this library applies it to
reflectometry values that are typically already normalized and are not raw counts. That distinction is important for
interpreting the fit behavior.

## Formula From The PDF

Section 3 of `mighell_fitting.pdf` defines the statistic

$$
\chi^2_\gamma = \sum_i \frac{[n_i + \min(n_i, 1) - m_i]^2}{n_i + 1}
$$

where:

- `n_i` are Poisson-distributed observed counts
- `m_i` are model values

The paper proposes this in preference to the modified Neyman statistic for Poisson data.

## Implemented Form

The code in `src/easyreflectometry/fitting.py` implements the same objective in weighted least-squares form.

For `objective='mighell'`, it constructs:

$$
y_{\mathrm{eff},i} = y_i + \min(y_i, 1)
$$

and

$$
\sigma_i = \sqrt{y_i + 1}
$$

with weights

$$
w_i = \frac{1}{\sigma_i}
$$

so the minimized least-squares objective is

$$
\sum_i \left(\frac{y_{\mathrm{eff},i} - f_i}{\sigma_i}\right)^2
=
\sum_i \frac{[y_i + \min(y_i,1) - f_i]^2}{y_i + 1}
$$

which is exactly the same algebraic form as the paper's `chi^2_gamma` statistic.


This has been implemented:
- shifting the target by `min(y, 1)`
- using `sqrt(y + 1)` as the effective uncertainty
- minimizing the resulting weighted residual sum of squares

## Important Scope Limitation

The PDF is explicitly about Poisson-distributed count data.

In `reflectometry-lib`, `mighell` is being applied to reflectometry intensities / reflectivities, which are generally:

- normalized values rather than raw counts
- potentially processed quantities rather than direct Poisson observations
- not guaranteed to satisfy the assumptions used in the paper's derivation


## The Full Mighell Fit Can Look Worse

The poor visual agreement of the full `mighell` fit against the measured reflectivity is expected under this
implementation (see results of running `zero_variances_example.py or
notebooks\zero_variance_fitting.ipynb)

That is because `mighell` here is not only a reweighting of the residuals. It also changes the fitted target:

$$
y \rightarrow y + \min(y,1)
$$

For any point with `0 < y < 1`, this doubles the target value.

At low `Q`, where reflectivity values can still be relatively large, the transformed target can be substantially higher
than the measured reflectivity. The optimizer is therefore solving a different problem than “fit the plotted
reflectivity values as closely as possible”.

The effect:

- `legacy_mask` and `hybrid` often nearly overlap
- full `mighell` can visibly deviate from the experimental curve
- the `mighell` objective value can be small while the classical chi-square is poor

## Hybrid mode

The `hybrid` mode is not taken directly from the PDF.

It is a project-specific adaptation that:

- uses standard weighted least squares where variances are valid
- applies the Mighell substitution only where variance is zero or invalid

This makes `hybrid` a better (?) reflectometry-oriented compromise rather than a direct reproduction of the paper.


I would propose using `hybrid` as default, but give users an option of both masking the zero variance points and applying proper Mighell algorithm.

