# EasyReflectometryLib Migration to New Corelib Architecture

## Known Issues (RESOLVED)

### test_copy & test_dict_round_trip numerical accuracy ✅ FIXED
- **Symptoms**: After copying or deserializing a Model with SurfactantLayer, reflectivity values differed (54.90 vs 51.23)
- **Root Cause**: Pre-existing serialization bug in `LayerAreaPerMolecule.as_dict()` - the `molecular_formula` attribute was not included in the serialization. When deserializing, the default formula was used instead of the actual formula, leading to incorrect scattering length and SLD calculations.
- **Fix Applied**: Added `this_dict['molecular_formula'] = self._molecular_formula` to `LayerAreaPerMolecule.as_dict()`
- **File**: `src/easyreflectometry/sample/elements/layers/layer_area_per_molecule.py`
- **Note**: This was a **pre-existing bug**, not caused by the corelib migration.
- **Status**: ✅ Fixed - All 441 tests now pass

---

## Overview

This document outlines the migration strategy for updating EasyReflectometryLib to use the new corelib module architecture.

## Core API Changes

| OLD corelib | NEW corelib |
|-------------|-------------|
| `ObjBase` | `ModelBase` |
| `CollectionBase` | `ModelCollection` |
| `InterfaceFactoryTemplate` | `CalculatorFactoryBase` |
| `SerializerComponent` | `CalculatorBase` (corelib) |

## Architectural Changes

- `CalculatorBase` now derives from `ModelBase`
- Calculator ownership moved from individual objects to `Project` class
- Individual objects no longer hold `interface` property
- Calculators follow stateful pattern: `Calculator(model, instrumental_parameters)`

## Migration PRs

### PR1: Base Classes Migration ✅ COMPLETE
**Scope:** Update base class inheritance while keeping interface property temporarily

**Files:**
- `src/easyreflectometry/sample/base_core.py` - `ObjBase` → `ModelBase`
- `src/easyreflectometry/sample/collections/base_collection.py` - `CollectionBase` → `ModelCollection`
- `src/easyreflectometry/model/model.py` - `ObjBase` → `ModelBase`
- All element classes (adjust constructors if needed)

**Testing:** 441 passed (after serialization bug fix)

### PR2: Calculator Refactor ✅ COMPLETE
**Scope:** Update calculator architecture to new pattern

**Files Modified:**
- `src/easyreflectometry/calculators/calculator_base.py` - Rewritten to:
  - Removed `SerializerComponent` inheritance
  - Added optional `model` parameter to constructor
  - Added `set_model()` method for stateful binding
  - Added `_create_all_bindings()` for full model hierarchy binding
  - Added `calculate()` method that uses the bound model
  - Added `reflectivity_profile()` (fixed typo from `reflectity_profile`)
  - Kept backwards compatible `reflectity_profile()` as alias
  - Added `fit_func` property for fitting framework compatibility
  
- `src/easyreflectometry/calculators/factory.py` - Rewritten to:
  - Inherits from `CalculatorFactoryBase` instead of `InterfaceFactoryTemplate`
  - Implements new abstract methods: `available_calculators`, `create()`
  - Maintains backwards compatibility with `available_interfaces`, `current_interface_name`
  - Keeps `__call__()` returning current calculator for existing code
  - Adds `generate_bindings()` method for backwards compatibility
  
- `src/easyreflectometry/calculators/refnx/calculator.py` - Updated to:
  - Accept optional `model` parameter in constructor
  - Initialize wrapper before calling super().__init__()
  
- `src/easyreflectometry/calculators/refl1d/calculator.py` - Updated to:
  - Accept optional `model` parameter in constructor
  - Initialize wrapper before calling super().__init__()

**Testing:** 
- All 441 tests pass
- Bug fix: `LayerAreaPerMolecule.as_dict()` now includes `molecular_formula` (pre-existing bug)

### PR3: Interface Removal ✅ COMPLETE
**Scope:** Remove distributed interface pattern, centralize calculator binding

**Architectural Change:**
- Sample objects (Material, Layer, Multilayer, etc.) no longer create bindings when `interface=` is passed
- Only `Model` triggers binding generation when interface is set
- Bindings are regenerated (not incrementally updated) when sample structure changes
- Calculator binding uses `set_model()` which properly traverses the model hierarchy

**Files Modified:**
- `src/easyreflectometry/sample/base_core.py`:
  - Made `interface` parameter optional with default `None`
  - Made `interface` property setter a no-op (stores value but doesn't propagate or trigger bindings)
  - Made `generate_bindings()` a no-op with deprecation docs
  
- `src/easyreflectometry/model/model.py`:
  - Constructor now triggers `generate_bindings()` when interface is passed (Model is top-level)
  - `add_assemblies()`, `duplicate_assembly()`, `remove_assembly()` now call `generate_bindings()` instead of incremental updates
  
- `src/easyreflectometry/calculators/factory.py`:
  - `generate_bindings()` now uses `set_model()` for proper hierarchy traversal (materials → layers → assemblies → model)

**Tests Updated:**
- `tests/sample/assemblies/test_multilayer.py` - 3 tests updated to use Model-based pattern
- `tests/sample/assemblies/test_repeating_multilayer.py` - 3 tests updated to use Model-based pattern
- `tests/sample/elements/layers/test_layer.py` - 1 test updated to use Model-based pattern

**Testing:** 441 passed, 5 skipped

## Import Changes

```python
# OLD
from easyscience import ObjBase as BaseObj
from easyscience.base_classes import CollectionBase as EasyBaseCollection
from easyscience.fitting.calculators.interface_factory import InterfaceFactoryTemplate
from easyscience.io import SerializerComponent

# NEW
from easyscience.base_classes import ModelBase
from easyscience.base_classes import ModelCollection
from easyscience.fitting.calculators import CalculatorFactoryBase
from easyscience.fitting.calculators import CalculatorBase
```

## Breaking Changes

### Removed
- Interface propagation through object hierarchy (sample objects no longer propagate interface to children)
- `generate_bindings()` automatic calls from individual sample objects
- Incremental binding updates (add_item_to_model, remove_item_from_model called directly)

### Changed
- `interface=` constructor parameter still accepted but is a no-op for sample objects
- `.interface` property still exists but setter is a no-op for sample objects (except Model)
- For `Model`, setting interface triggers `generate_bindings()` (backward compatible)
- `CalculatorFactory` now inherits `CalculatorFactoryBase`
- Calculators are stateful (hold model reference)
- `generate_bindings()` now uses `set_model()` for proper hierarchy traversal
- Sample structure changes trigger full binding regeneration, not incremental updates

## Testing Strategy

Migration tests will be added to `tests/test_migration.py` and can be removed after migration is complete.

## Decisions Made

1. **Backwards compatibility:** Not required - moving away from old API
2. **Calculator sharing:** Multiple models can share one Calculator via MultiFitter
3. **Migration tests:** Separate file (`tests/test_migration.py`)
4. **Calculator pattern:** Stateful pattern matching new corelib `CalculatorBase`
