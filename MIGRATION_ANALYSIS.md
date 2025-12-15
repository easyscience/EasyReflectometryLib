# EasyReflectometryLib Migration Analysis - Phase 1

## Executive Summary

This document contains the impact analysis for migrating EasyReflectometryLib from the old corelib architecture (`ObjBase`, `CollectionBase`) to the new architecture (`ModelBase`, `ModelCollection`, `CalculatorFactoryBase`).

---

## 1. Inheritance Tree Analysis

### Current Architecture (Before Migration)

```
ObjBase (corelib - deprecated)
├── BaseCore (easyreflectometry/sample/base_core.py)
│   ├── Material
│   │   ├── MaterialDensity
│   │   ├── MaterialMixture
│   │   └── MaterialSolvated
│   ├── Layer
│   │   └── LayerAreaPerMolecule
│   ├── BaseAssembly
│   │   ├── Multilayer
│   │   ├── RepeatingMultilayer
│   │   ├── SurfactantLayer
│   │   └── GradientLayer
│   └── Model (easyreflectometry/model/model.py)

CollectionBase (corelib - deprecated)
├── BaseCollection (easyreflectometry/sample/collections/base_collection.py)
│   ├── LayerCollection
│   ├── MaterialCollection
│   └── Sample

InterfaceFactoryTemplate (corelib - deprecated)
└── CalculatorFactory (easyreflectometry/calculators/factory.py)
```

### Target Architecture (After Migration)

```
ModelBase (corelib - new)
├── BaseCore (with compatibility layer)
│   ├── Material
│   │   ├── MaterialDensity
│   │   ├── MaterialMixture
│   │   └── MaterialSolvated
│   ├── Layer
│   │   └── LayerAreaPerMolecule
│   ├── BaseAssembly
│   │   ├── Multilayer
│   │   ├── RepeatingMultilayer
│   │   ├── SurfactantLayer
│   │   └── GradientLayer
│   └── Model

ModelCollection (corelib - new)
├── BaseCollection (with compatibility layer)
│   ├── LayerCollection
│   ├── MaterialCollection
│   └── Sample

CalculatorFactoryBase (corelib - new)
└── CalculatorFactory
```

---

## 2. Calculator Ownership Migration

### Current Pattern (Interface on Each Object)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Current Design                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Model ──────────────► interface ◄──── CalculatorFactory       │
│     │                                          │                │
│     ▼                                          │                │
│   Sample ─────────────► interface ◄────────────┘                │
│     │                                          │                │
│     ▼                                          │                │
│   Multilayer ─────────► interface ◄────────────┘                │
│     │                                          │                │
│     ▼                                          │                │
│   Layer ──────────────► interface ◄────────────┘                │
│     │                                          │                │
│     ▼                                          │                │
│   Material ───────────► interface ◄────────────┘                │
│                                                                 │
│   Every object holds a reference to the same interface          │
│   Interface propagates through generate_bindings()              │
└─────────────────────────────────────────────────────────────────┘
```

### Current Pattern (After PR3 - Centralized Calculator Ownership) ✅ IMPLEMENTED

```
┌─────────────────────────────────────────────────────────────────┐
│                    Implemented Design (PR3)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Model(interface=factory) ──────► CalculatorFactory            │
│     │                                    │                      │
│     │ (triggers generate_bindings)       │                      │
│     │                                    ▼                      │
│     │                            factory.set_model(model)       │
│     │                                    │                      │
│     ▼                                    ▼                      │
│   Sample                         _create_all_bindings()         │
│     │                              (materials → layers →        │
│     ▼                               assemblies → model)         │
│   Multilayer                                                    │
│     │                                                           │
│     ▼                                                           │
│   Layer                                                         │
│     │                                                           │
│     ▼                                                           │
│   Material                                                      │
│                                                                 │
│   Only Model triggers binding generation                        │
│   Sample objects store interface but don't propagate/bind       │
│   Calculator walks model hierarchy via set_model()              │
│   Structure changes trigger full binding regeneration           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Breaking Changes Inventory

| Component | Old API | New API | Breaking Change | Mitigation |
|-----------|---------|---------|-----------------|------------|
| `BaseCore` | Inherits `ObjBase` | Inherits `ModelBase` | Constructor signature | Add compatibility `__init__` |
| `BaseCore.name` | Direct property | Maps to `display_name` | Property access | Property wrapper |
| `BaseCore.interface` | Direct property | To be removed (PR3) | Property access | Keep temporarily |
| `BaseCollection` | Inherits `CollectionBase` | Inherits `ModelCollection` | Constructor signature | Add compatibility `__init__` |
| `Model` | Inherits `ObjBase` | Inherits `ModelBase` | Constructor signature | Explicit properties |
| `CalculatorFactory` | Inherits `InterfaceFactoryTemplate` | Inherits `CalculatorFactoryBase` | Factory pattern | Adapter pattern |
| Serialization | `as_dict()` custom | `to_dict()` from SerializerBase | Method names | Alias methods |
| Object creation | `name` parameter | `unique_name` + `display_name` | Parameter naming | Map in constructor |
| Parameter access | `self.kwarg_name` magic | Explicit properties | Attribute access | Custom `__getattr__` |

---

## 4. Migration Strategy Assessment

| Aspect | Assessment | Risk Level | Notes |
|--------|------------|------------|-------|
| Backward Compatibility | Not required | Low | User confirmed no BC needed |
| Incremental Approach | ✅ Completed | Low | 3 PRs successfully isolated changes |
| Test Coverage | Good (441 tests) | Low | All tests pass after migration |
| Calculator Refactor | ✅ Complete | Resolved | PR2 successfully refactored |
| Interface Removal | ✅ Complete | Resolved | PR3 centralized binding |
| Serialization | ✅ Fixed | Resolved | molecular_formula bug fixed |
| Parameter Dependencies | Working | Low | SurfactantLayer tested and working |
| Global Object Map | Working | Low | No issues encountered |

**Migration Status: ✅ COMPLETE**
- All 3 PRs implemented
- All 441 tests pass
- Architecture migrated from distributed interface to centralized calculator binding

---

## 5. Pull Request Strategy

### PR1: Base Class Replacements (Current)

| Task | Status | Files Modified |
|------|--------|----------------|
| `BaseCore`: `ObjBase` → `ModelBase` | ✅ Complete | `sample/base_core.py` |
| `BaseCollection`: `CollectionBase` → `ModelCollection` | ✅ Complete | `sample/collections/base_collection.py` |
| `Model`: Migrate to `ModelBase` | ✅ Complete | `model/model.py` |
| Add `name` property (→ `display_name`) | ✅ Complete | `sample/base_core.py` |
| Add `__getattr__`/`__setattr__` for kwargs | ✅ Complete | `sample/base_core.py` |
| Custom `as_dict`/`to_dict`/`from_dict` | ✅ Complete | Multiple files |
| Fix test failures | ✅ Complete | Multiple files |
| **Test Results** | **441 passed (after bug fix)** | |

### PR2: Calculator Refactor (Complete)

| Task | Status | Files Modified |
|------|--------|----------------|
| `CalculatorFactory`: `InterfaceFactoryTemplate` → `CalculatorFactoryBase` | ✅ Complete | `calculators/factory.py` |
| Refactor `CalculatorBase` to stateful pattern | ✅ Complete | `calculators/calculator_base.py` |
| Add optional `model` parameter to calculators | ✅ Complete | `calculators/refnx/calculator.py`, `calculators/refl1d/calculator.py` |
| Add `set_model()` for stateful binding | ✅ Complete | `calculators/calculator_base.py` |
| Add `_create_all_bindings()` for model hierarchy | ✅ Complete | `calculators/calculator_base.py` |
| Add `calculate()` method using bound model | ✅ Complete | `calculators/calculator_base.py` |
| Maintain backwards compatibility in factory | ✅ Complete | `calculators/factory.py` |
| Fix pre-existing serialization bug | ✅ Complete | `layer_area_per_molecule.py` |
| **Test Results** | **441 passed** | |

### PR3: Interface Removal ✅ COMPLETE

| Task | Status | Files Modified |
|------|--------|----------------|
| Make `interface` a no-op on sample objects | ✅ Complete | `sample/base_core.py` |
| Make `generate_bindings` a no-op on sample objects | ✅ Complete | `sample/base_core.py` |
| Keep `interface` property for backward compat (but no-op) | ✅ Complete | `sample/base_core.py` |
| Update Model to trigger bindings when interface set | ✅ Complete | `model/model.py` |
| Update Model methods to regenerate bindings | ✅ Complete | `model/model.py` |
| Update factory.generate_bindings to use set_model | ✅ Complete | `calculators/factory.py` |
| Update tests to use Model-based pattern | ✅ Complete | `test_multilayer.py`, `test_repeating_multilayer.py`, `test_layer.py` |
| **Test Results** | **441 passed** | |

---

## 6. Files Affected by Migration

### Core Files (Modified in PR1)

| File | Changes Made |
|------|--------------|
| `src/easyreflectometry/sample/base_core.py` | Complete rewrite with ModelBase |
| `src/easyreflectometry/sample/collections/base_collection.py` | ModelCollection inheritance |
| `src/easyreflectometry/model/model.py` | ModelBase inheritance, explicit properties |
| `src/easyreflectometry/project.py` | Fixed material duplication bug |
| `src/easyreflectometry/sample/collections/layer_collection.py` | Fixed `layers` kwarg handling |
| `src/easyreflectometry/sample/collections/sample.py` | Fixed isinstance order |
| `tests/conftest.py` | New - pytest fixture for global_object cleanup |
| `tests/test_project.py` | Updated test expectation |

### Files with Minor Changes

| File | Changes Made |
|------|--------------|
| `src/easyreflectometry/sample/assemblies/gradient_layer.py` | Safe dict key removal |
| `src/easyreflectometry/sample/assemblies/surfactant_layer.py` | Safe dict key removal |
| `src/easyreflectometry/sample/elements/layers/layer_area_per_molecule.py` | Safe dict key removal |
| `src/easyreflectometry/sample/elements/materials/material_density.py` | Safe dict key removal |
| `src/easyreflectometry/sample/elements/materials/material_solvated.py` | Safe dict key removal |

### Calculator Files (Modified in PR2)

| File | Changes Made |
|------|--------------|
| `src/easyreflectometry/calculators/calculator_base.py` | Complete rewrite: removed `SerializerComponent` inheritance, added stateful model binding, `set_model()`, `_create_all_bindings()`, `calculate()`, `reflectivity_profile()`, `fit_func` property |
| `src/easyreflectometry/calculators/factory.py` | Inherits `CalculatorFactoryBase`, implements `available_calculators`, `create()`, maintains backwards compatibility with `__call__()`, `current_interface_name`, `generate_bindings()` |
| `src/easyreflectometry/calculators/refnx/calculator.py` | Accept optional `model` parameter, initialize wrapper before `super().__init__()` |
| `src/easyreflectometry/calculators/refl1d/calculator.py` | Accept optional `model` parameter, initialize wrapper before `super().__init__()` |

### Interface Removal Files (Modified in PR3)

| File | Changes Made |
|------|--------------|
| `src/easyreflectometry/sample/base_core.py` | Made `interface` parameter optional with `None` default, made `interface` setter a no-op (stores but doesn't propagate), made `generate_bindings()` a no-op |
| `src/easyreflectometry/model/model.py` | Constructor triggers `generate_bindings()` when interface passed, `add_assemblies()`, `duplicate_assembly()`, `remove_assembly()` call `generate_bindings()` instead of incremental updates |
| `src/easyreflectometry/calculators/factory.py` | `generate_bindings()` now uses `set_model()` for proper hierarchy traversal |
| `tests/sample/assemblies/test_multilayer.py` | Updated 3 tests to use Model-based binding pattern |
| `tests/sample/assemblies/test_repeating_multilayer.py` | Updated 3 tests to use Model-based binding pattern |
| `tests/sample/elements/layers/test_layer.py` | Updated 1 test to use Model-based binding pattern |

---

## 7. Known Issues (Resolved)

### Numerical Accuracy After Copy/Deserialize (FIXED)

**Affected Tests:**
- `tests/test_topmost_nesting.py::test_copy`
- `tests/model/test_model.py::test_dict_round_trip[interface1]`

**Symptoms:**
- After copying or deserializing a Model with SurfactantLayer, reflectivity values differed
- Original: 54.90, Copy: 51.23 (difference: 3.67)

**Root Cause (Pre-existing Bug):**
The `LayerAreaPerMolecule.as_dict()` method was missing the `molecular_formula` attribute in its serialization output. This caused:
1. When deserializing, a new `LayerAreaPerMolecule` was created with the **default** molecular formula (`C10H18NO8P`) instead of the actual formula (e.g., `C32D64` for DPPC tail)
2. The scattering length was recomputed from the wrong formula
3. The SLD (which depends on scattering length via a Parameter dependency expression) was incorrect
4. This resulted in different reflectivity calculations

**Fix Applied:**
Added `molecular_formula` to the `as_dict()` output in `layer_area_per_molecule.py`:
```python
this_dict['molecular_formula'] = self._molecular_formula
```

**Note:** This was a **pre-existing bug** in EasyReflectometryLib, not caused by the corelib migration. It was discovered during PR2 testing because `test_copy` and `test_dict_round_trip` exercise the serialization path for SurfactantLayer.

**Status:** ✅ Fixed

---

## 8. Corelib Changes Required

The following change was made to corelib during this migration:

**File:** `corelib/src/easyscience/base_classes/collection_base.py`

**Change:** Added `NewBase` to accepted types in `CollectionBase.__init__`

```python
# Before
if not isinstance(item, (BasedBase, DescriptorBase)):
    raise TypeError(...)

# After  
if not isinstance(item, (BasedBase, DescriptorBase, NewBase)):
    raise TypeError(...)
```

This allows `ModelCollection` (which inherits from `CollectionBase`) to accept objects that inherit from `ModelBase` (which inherits from `NewBase`).
