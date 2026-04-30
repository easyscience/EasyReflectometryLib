# Version 1.6.0 (1 May 2026)

Add Mighell-based handling of non-positive-variance points in fitting (issue #256).
Non-positive-variance data points are no longer forcibly discarded; instead, a
hybrid objective applies a Mighell substitution for non-positive-variance points
while using standard weighted least squares for the rest. The previous masking
behavior is available via `objective='legacy_mask'`. New `objective` parameter on
`MultiFitter`, `fit()`, and `fit_single_data_set_1d()`.

# Version 1.3.3 (17 June 2025)

Added Chi^2 and fit status to fitting results.
Added explicit dependency on bumps version.
