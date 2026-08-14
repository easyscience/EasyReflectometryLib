# Unreleased

All four polarization channels (pp, pm, mp, mm) are now available from
the refl1d calculator; previously only the non-spin-flip pp channel was
returned.

- New `LayerMagnetism` sample element makes magnetism part of the model:
  `Layer` accepts an optional `magnetism` (with `rho_m`, the magnetic
  SLD, and `theta_m`, the in-plane moment angle, as fittable, serialized
  `Parameter`s). Attaching a magnetic layer automatically enables
  `include_magnetism` on the calculator (raising `NotImplementedError`
  on backends without magnetism support); removing the last magnetic
  layer disables it again. `Model.has_magnetism`,
  `CalculatorBase.supports_magnetism` and
  `Project.calculator_supports_magnetism` expose the state to
  applications.
- New `PolarizedDataSet` groups per-spin-channel `DataSet1D` objects
  (one file per channel; 'pp'/'mm' only for NSF experiments, spin-flip
  channels optional) into one experiment sharing a single model.
  `Project.load_polarized_experiment(paths)` loads it from an explicit
  channel → file mapping, and
  `Project.suggest_polarized_channel_assignment(paths)` pre-fills that
  mapping from the ORSO header polarization (`pp`/`mm`/`pm`/`mp` only —
  partially-analysed observables such as `po`/`mo`, which measure
  channel sums, and `op`/`om`/`unpolarized` are left for the user to
  decide) or, for plain text files, from filename tokens
  (`_uu`/`_up`/`_pp` → pp, `_dd`/`_down`/`_mm` → mm, `_ud`/`_pm` → pm,
  `_du`/`_mp` → mp).
- Experiment and model accessors are channel aware:
  `Project.experimental_data_for_model_at_index(index, channel=...)`
  returns the `DataSet1D` of one spin channel (`None`, the default,
  keeps the previous behavior and returns the stored experiment),
  `Project.model_data_for_model_at_index(index, q_range, channel=...)`
  calculates one spin cross-section, and
  `Project.experiment_is_polarized_at_index(index)` /
  `Project.experiment_channels_at_index(index)` report the polarization
  state. Asking for a channel that was not measured raises `KeyError`;
  an unknown channel, or any channel on an unpolarized experiment,
  raises `ValueError`.
- The summary/report figures now show one measured series per spin
  channel of a polarized experiment, each in its channel color, with the
  matching calculated cross-section. Channels whose cross-section cannot
  be calculated (e.g. spin-flip on a non-magnetic model) are shown
  without a calculated overlay rather than with the channel-agnostic
  curve. Previously a polarized experiment made the report figures fail
  on `PolarizedDataSet.x`.
- New `calculate_channel(q, model, channel)` on the wrapper (and
  `reflectivity_profile_channel` on the calculator,
  `fit_func_for_channel` on `CalculatorFactory`) evaluates one explicit
  spin channel without touching the global `polarization_channel` state.
- New `MultiFitter.fit_polarized(data)` fits all measured channels of a
  `PolarizedDataSet` simultaneously against the shared model: one fit
  function per channel, common structural parameters, magnetic
  parameters constrained by all channels at once. Returns per-channel
  `FitResults`.
- The refl1d wrapper now caches the four polarized cross-sections per
  model state and (q, dq) grid — they come from a single kernel
  evaluation, so a simultaneous N-channel fit costs about one evaluation
  per iteration instead of N.

- New `polarized_reflectivity_profiles(x_array, model_id)` on the
  calculator (and on `CalculatorFactory`) returns the reflectivity of
  all four spin channels in one calculation as a dictionary keyed
  `'pp'`, `'pm'`, `'mp'`, `'mm'` (in that order). Requires
  `include_magnetism = True`.
- New `polarization_channel` property (accepts
  `'pp'`/`'pm'`/`'mp'`/`'mm'` or the new `PolarizationChannel` enum)
  selects which channel `reflectity_profile` — and hence fitting —
  returns, enabling fits against spin-flip or mm data. Default `'pp'`;
  disabling magnetism resets it to `'pp'`. Note: the channel belongs to
  the currently active calculator instance, not to a model or dataset —
  it affects every subsequent calculation with that calculator, and
  `interface.switch(...)` constructs a fresh calculator, resetting it
  (along with `include_magnetism`).
- New `magnetic_sld_profile(model_id)` on the calculator (and on
  `CalculatorFactory`) returns the nuclear and magnetic scattering
  length density profiles as a tuple `z`, `sld(z)`, `rhoM(z)` (magnetic
  SLD) and `thetaM(z)` (magnetic angle). Requires
  `include_magnetism = True`; refl1d only.
- Magnetic calculations now always build all four refl1d cross-sections,
  so they may take somewhat longer than before; pp results are
  unchanged.
- Bug fix: `include_magnetism = True` on a refnx-backed calculator now
  raises `NotImplementedError`. Previously it was silently accepted (the
  guard sat on a property the calculator never called) even though refnx
  magnetism is not supported.
- Bug fix (pre-existing): disabling magnetism after layers were created
  with it enabled used to leave refl1d `Magnetism` objects on the slabs,
  making a subsequent unpolarized calculation raise `AttributeError`
  inside refl1d. Disabling magnetism now strips the magnetic state from
  existing layers, so the unpolarized path works again. Magnetic
  parameters (`rhoM`/`thetaM`) are kept in a per-layer store inside the
  wrapper, so they survive a disable/re-enable cycle and are re-attached
  when magnetism is enabled again; `update_layer` also accepts the
  magnetism keys one at a time.

# Version 1.7.0 (1 Aug 2026)

Restored the measured per-point resolution on data load (issue #368).

- Loading data through `Project` (`load_new_experiment`,
  `load_experiment_for_model_at_index`,
  `load_all_experiments_from_file`) again sets a `Pointwise` resolution
  function when the file carries per-point q-resolution (an sQz column
  in `.ort` files, or a 4th column in text files). Since PR #293 the
  loaders discarded this data and always applied a flat
  `PercentageFwhm(5.0)` — a temporary workaround that never got
  reverted. Fits of such data were smeared with 5% FWHM regardless of
  what the instrument delivered and should be re-run.
- Files without q-resolution data keep the 5% FWHM default. The pre-#293
  fallback that built a `LinearSpline` from the _reflectivity_ error
  (`sqrt(ye)`) was not restored: a reflectivity uncertainty is not a
  q-width, and that branch produced effectively zero smearing.
- Known limitation (pre-existing): the resolution function lives on the
  model, so when several experiments share one model the last-loaded
  dataset's resolution wins.

Fixed inconsistent interpretation of vector resolution functions between
the refnx and refl1d engines (issue #367).

- **Reflectivity results change for two engine / resolution
  combinations.** `LinearSpline` on refl1d previously **over-smeared by
  a factor of 2.355** (its FWHM widths were passed to refl1d's
  `probe.dQ`, which expects sigma). `Pointwise` on refnx previously
  **under-smeared by the same factor** (its sigma widths were passed to
  refnx's `x_err`, which expects FWHM). Both are now correct. Fits and
  simulations that used either combination will produce different —
  previously wrong — results and should be re-run. `PercentageFwhm` on
  either engine, `LinearSpline` on refnx, and `Pointwise` on refl1d are
  numerically unchanged.
- `ResolutionFunction.smearing()` now returns **sigma** (the Gaussian
  standard deviation) for every subclass; each engine wrapper converts
  to its backend's convention. This is a behavioural change to a public
  method. Most visibly, `PercentageFwhm.smearing(q)` used to return the
  _percentage_ itself (e.g. `5.0`) and now returns an absolute sigma
  (e.g. `0.00212` at `q=0.1`); `LinearSpline.smearing(q)` returns its
  `fwhm_values` divided by `2*sqrt(2*ln2)`. Callers relying on the old
  values need to convert. The new `SIGMA_TO_FWHM` constant is exported
  from `easyreflectometry.model.resolution_functions`.
- Constructors are **unchanged**: `PercentageFwhm(5)` still means 5%
  FWHM and `LinearSpline(q, fwhm_values)` still takes FWHM. Only the
  `smearing()` output convention moved, so existing model-building code
  needs no edits.
- `PercentageFwhm.smearing(q)` given a scalar `q` now returns a 0-d
  numpy scalar rather than a shape-`(1,)` array, matching
  `LinearSpline`. `smearing(0.1)[0]` therefore raises `IndexError` where
  it previously returned a value.

Migrated sample / model classes off the deprecated `easyscience.ObjBase`
and `easyscience.CollectionBase` pipeline.

- `BaseCore` is now built on `ModelBase`; `BaseCollection` on
  `EasyList`. `Model`, `Material`, `Layer`, `MaterialMixture`,
  `MaterialSolvated`, `LayerAreaPerMolecule`, `Multilayer`,
  `RepeatingMultilayer`, `GradientLayer`, `Bilayer`, `SurfactantLayer`,
  `BaseAssembly`, `LayerCollection`, `MaterialCollection`, `Sample`, and
  `ModelCollection` were all rewritten to use the new bases.
- Properties returning a `Parameter` (`Material.sld`-style) now expose
  the `Parameter` object directly across all sample classes, replacing
  the inconsistent legacy behaviour where `MaterialMixture.fraction`,
  `MaterialSolvated.solvent_fraction`,
  `LayerAreaPerMolecule.area_per_molecule`, and
  `LayerAreaPerMolecule.solvent_fraction` returned `float`. Read the
  value via `.value` (e.g. `material_mixture.fraction.value`). Setters
  still accept a float. `MaterialMixture.sld` / `MaterialMixture.isld`
  remain `float` — they are derived via constraints, not constructor
  arguments.
- `BaseCollection.remove(index)` (the legacy index-based helper) renamed
  to `remove_at(index)`. The standard `MutableSequence.remove(value)` is
  now inherited unmodified.
- Project files saved by previous versions cannot be read.
  `Project.as_dict` writes `file_format=2`; `Project.from_dict` raises a
  clear `ValueError` on missing or unsupported markers.
- `model.get_parameters()` / `collection.get_parameters()` still work
  (kept as compatibility shims) but new code should use
  `get_all_parameters()`.
- No more `DeprecationWarning` from `easyscience.ObjBase` /
  `CollectionBase` on construction of any sample / model object.

# Version 1.6.0 (1 May 2026)

Add Mighell-based handling of non-positive-variance points in fitting
(issue #256). Non-positive-variance data points are no longer forcibly
discarded; instead, a hybrid objective applies a Mighell substitution
for non-positive-variance points while using standard weighted least
squares for the rest. The previous masking behavior is available via
`objective='legacy_mask'`. New `objective` parameter on `MultiFitter`,
`fit()`, and `fit_single_data_set_1d()`.

# Version 1.3.3 (17 June 2025)

Added Chi^2 and fit status to fitting results. Added explicit dependency
on bumps version.
