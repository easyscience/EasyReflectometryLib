# EasyReflectometryLib Migration to New Corelib Architecture

## Known Issues to Fix Later

### test_copy numerical accuracy (test_topmost_nesting.py)
- **Issue**: After copying a Model with SurfactantLayer, the reflectivity profile produces slightly different values (54.90 vs 51.23)
- **Cause**: Likely related to how calculator bindings or dependencies are restored during from_dict deserialization
- **Status**: Deferred - to be investigated after PR1 is complete

### test_dict_round_trip[interface1] numerical accuracy (test_model.py)
- **Issue**: Same as above - after from_dict, reflectivity values differ
- **Cause**: Same root cause - SurfactantLayer dependencies not properly restored
- **Status**: Deferred - same fix needed as test_copy

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

### PR1: Base Classes Migration
**Scope:** Update base class inheritance while keeping interface property temporarily

**Files:**
- `src/easyreflectometry/sample/base_core.py` - `ObjBase` → `ModelBase`
- `src/easyreflectometry/sample/collections/base_collection.py` - `CollectionBase` → `ModelCollection`
- `src/easyreflectometry/model/model.py` - `ObjBase` → `ModelBase`
- All element classes (adjust constructors if needed)

**Testing:** Existing tests should pass

### PR2: Calculator Refactor
**Scope:** Update calculator architecture to new pattern

**Files:**
- `src/easyreflectometry/calculators/calculator_base.py` - inherit from corelib `CalculatorBase`
- `src/easyreflectometry/calculators/factory.py` - `InterfaceFactoryTemplate` → `CalculatorFactoryBase`
- `src/easyreflectometry/calculators/refnx/calculator.py` - stateful pattern
- `src/easyreflectometry/calculators/refl1d/calculator.py` - stateful pattern

**Testing:** New tests for calculator binding

### PR3: Interface Removal
**Scope:** Remove interface from all classes, update Project

**Files:**
- `src/easyreflectometry/project.py` - new binding methods
- All sample classes - remove `interface` parameter and property
- All collection classes - remove `interface` parameter and property
- `src/easyreflectometry/model/model.py` - remove interface

**Testing:** All tests updated for new pattern

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
- `interface=` constructor parameter from all sample/model classes
- `.interface` property from all sample/model classes
- Interface propagation through object hierarchy
- `generate_bindings()` calls from individual objects

### Changed
- `CalculatorFactory` now inherits `CalculatorFactoryBase`
- Calculators are stateful (hold model reference)
- `Project` owns calculator binding lifecycle

## Testing Strategy

Migration tests will be added to `tests/test_migration.py` and can be removed after migration is complete.

## Decisions Made

1. **Backwards compatibility:** Not required - moving away from old API
2. **Calculator sharing:** Multiple models can share one Calculator via MultiFitter
3. **Migration tests:** Separate file (`tests/test_migration.py`)
4. **Calculator pattern:** Stateful pattern matching new corelib `CalculatorBase`
