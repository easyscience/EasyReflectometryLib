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

### Target Pattern (Centralized Calculator Ownership)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Target Design                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Project ─────────────────────────► Calculator                 │
│     │                                    │                      │
│     │                                    │                      │
│     ▼                                    ▼                      │
│   Model ◄────────────────────── Calculator.model_ref            │
│     │                                                           │
│     ▼                                                           │
│   Sample                                                        │
│     │                                                           │
│     ▼                                                           │
│   Multilayer                                                    │
│     │                                                           │
│     ▼                                                           │
│   Layer                                                         │
│     │                                                           │
│     ▼                                                           │
│   Material                                                      │
│                                                                 │
│   Only Project owns the Calculator                              │
│   Calculator holds reference to Model for calculations          │
│   Sample objects have NO interface property                     │
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
| Incremental Approach | Recommended | Medium | 3 PRs to isolate changes |
| Test Coverage | Good (446 tests) | Low | Comprehensive test suite exists |
| Calculator Refactor | Complex | High | Touches many files, defer to PR2/PR3 |
| Serialization | Medium complexity | Medium | Need custom `as_dict`/`from_dict` |
| Parameter Dependencies | Complex | High | SurfactantLayer uses complex dependencies |
| Global Object Map | Requires care | Medium | Name collisions possible |

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
| **Test Results** | **439 passed, 2 deferred** | |

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
| **Test Results** | **415 passed, 2 deferred (same as PR1)** | |

### PR3: Interface Removal (Planned)

| Task | Status | Files to Modify |
|------|--------|-----------------|
| Remove `interface` property from sample objects | Planned | All sample classes |
| Remove `generate_bindings` from sample objects | Planned | `base_core.py` |
| Update all constructors to not accept `interface` | Planned | All sample classes |
| Clean up interface propagation code | Planned | Multiple |

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

---

## 7. Known Issues (Deferred)

### Numerical Accuracy After Copy/Deserialize

**Affected Tests:**
- `tests/test_topmost_nesting.py::test_copy`
- `tests/model/test_model.py::test_dict_round_trip[interface1]`

**Symptoms:**
- After copying or deserializing a Model with SurfactantLayer, reflectivity values differ
- Original: 54.90, Copy: 51.23 (difference: 3.67)

**Likely Cause:**
- SurfactantLayer uses complex Parameter dependencies
- Dependencies may not be properly restored during deserialization
- Calculator bindings may differ between original and restored objects

**Status:** Deferred to post-PR1 investigation

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
