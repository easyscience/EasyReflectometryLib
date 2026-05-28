# Unreleased

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
