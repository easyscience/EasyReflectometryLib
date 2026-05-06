# Bayesian Analysis in EasyReflectometryLib — Implementation Plan

## Current State

- `BAYESIAN_BUMPS.md` (in `MD/`) describes a 3-phase plan. Phase 1 (docs/notebook) is complete.
- `bayesian_bumps.py` (attached notebook) works but users must directly import from `bumps.fitters`, `bumps.names`, `bumps.parameter` — a leaky abstraction.
- The `Bumps` minimizer in `easyscience` (`core/src/easyscience/fitting/minimizers/minimizer_bumps.py`) only exposes **classical optimization** methods (`amoeba`, `newton`, `lm`), NOT the DREAM/MCMC sampler.
- The `AvailableMinimizers` enum only has `Bumps`, `Bumps_simplex`, `Bumps_newton`, `Bumps_lm`. This should remain optimizer-focused; DREAM should be exposed as a sampling workflow, not as another minimizer choice.
- `MultiFitter` in reflectometry-lib currently only has `fit()` and `fit_single_data_set_1d()`.

## Goal

Users should be able to run Bayesian MCMC sampling with a **clean high-level API** like:

```python
fitter = MultiFitter(model)
fitter.switch_minimizer(AvailableMinimizers.Bumps)

# Classical fit first
analysed = fitter.fit(data)

# Bayesian sampling
posterior = fitter.sample(data, samples=5000, burn=1000, thin=10)

# Analyze
from easyreflectometry.analysis.bayesian import plot_corner, posterior_summary
plot_corner(posterior)
print(posterior_summary(posterior))
```

Important API boundary: `fit()` remains classical optimization only. Bayesian DREAM sampling is exposed through `sample()` so users do not receive sampler-shaped results from an optimizer-shaped API.

## Implementation Plan

### Step 1 — Keep DREAM separate from `AvailableMinimizers`

**File**: `core/src/easyscience/fitting/available_minimizers.py`

Do **not** add `Bumps_dream` as a normal `AvailableMinimizers` member. The enum is currently used to instantiate minimizer backends and to route calls through `Fitter.fit()`, which expects optimizer-style `FitResults`. DREAM is an MCMC sampler and returns a sampler state/chain rather than a best-fit result.

Use `AvailableMinimizers.Bumps` to select the BUMPS backend, then expose DREAM through a dedicated `sample()` method. This avoids making `project.minimizer = AvailableMinimizers.Bumps_dream` look like a valid classical fitting mode.

### Step 2 — Add dedicated DREAM sampling support to the Bumps minimizer (core repo)

**File**: `core/src/easyscience/fitting/minimizers/minimizer_bumps.py`

**2a.** Keep `supported_methods()` optimizer-only (`amoeba`, `newton`, `lm`). Do not add `'dream'` there unless the EasyScience fitting abstraction is later split into optimizer and sampler concepts.

**2b.** Add a new method `sample()` to the `Bumps` class:

```python
def sample(
    self,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    samples: int = 10000,
    burn: int = 2000,
    thin: int = 10,
    chains: int | None = None,
    population: int | None = None,
    model: Callable | None = None,
    parameters: list | None = None,
    progress_callback: Callable | None = None,
    seed: int | None = None,
    **kwargs,
) -> dict:
    """Run Bayesian MCMC sampling using BUMPS DREAM sampler.

    Returns a dict with:
      - 'draws': np.ndarray, shape (n_samples, n_params) — posterior samples
      - 'param_names': list[str] — parameter names
      - 'state': DreamState — raw BUMPS state for save/restore
      - 'logp': np.ndarray — log-posterior values
    """
```

The implementation would:
1. Build the `Curve` model + `FitProblem` (reuse `_make_model()` logic)
2. Translate user-friendly aliases to BUMPS DREAM settings and call `bumps_fit(problem, method='dream', samples=samples, burn=burn, thin=thin, pop=population, ...)`
3. Extract `result.state.draw().points` and return structured dict
4. Preserve and restore the EasyScience global object stack state, mirroring the existing `fit()` implementation
5. Handle multi-dataset via the same `MultiFitter._precompute_reshaping` pattern

**2c.** Do **not** modify `fit()` to handle `method='dream'`, and do **not** delegate `fit(method='dream')` to `sample()`. `fit()` returns `FitResults`; `sample()` returns posterior samples and sampler metadata. Keeping the methods separate prevents incompatible return types from leaking into `Fitter.fit()` and reflectometry-lib `MultiFitter.fit()`.

Recommended alias mapping:

| Public argument | BUMPS DREAM setting | Rationale |
|-----------------|---------------------|-----------|
| `samples` | `samples` | Clear user-facing chain length; prefer over optimizer-oriented `steps`. |
| `burn` | `burn` | Matches BUMPS and common MCMC terminology. |
| `thin` | `thin` | Matches BUMPS and common MCMC terminology. |
| `chains` | `pop` | User-friendly MCMC wording; maps to BUMPS population count. |
| `population` | `pop` | BUMPS-aware alias for advanced users. |
| `initialization` | `init` | More readable than `init`, but pass through to BUMPS. |
| `seed` | RNG seeding before sampling | Expose reproducibility without requiring users to know BUMPS internals. |

If both `chains` and `population` are provided, raise `ValueError` unless they match. Accept `steps` only as a deprecated alias for `samples`, with a warning, because `steps` already means optimizer budget in EasyScience `Bumps.fit()`.

### Step 3 — Add `sample()` to reflectometry-lib `MultiFitter`

**File**: `reflectometry-lib/src/easyreflectometry/fitting.py`

Add a `sample()` method to `MultiFitter`:

```python
def sample(
    self,
    data: sc.DataGroup,
    samples: int = 10000,
    burn: int = 2000,
    thin: int = 10,
    chains: int | None = None,
    population: int | None = None,
    seed: int | None = None,
    objective: str | None = None,
) -> dict:
    """Run Bayesian MCMC sampling on reflectometry data.

    :param data: DataGroup with reflectivity data.
    :param samples: Number of retained DREAM samples requested from BUMPS.
    :param burn: Burn-in steps.
    :param thin: Thinning interval.
    :param chains: User-friendly alias for BUMPS DREAM population count.
    :param population: BUMPS DREAM population count (`pop`) for advanced users.
    :param seed: Random seed for reproducibility.
    :param objective: Zero-variance handling strategy.
    :return: Dict with posterior samples, parameter names, and sampler state.
    """
```

Internally:
1. Reuse `_prepare_fit_arrays` for data preparation
2. Mirror the EasyScience `Fitter.fit()` lifecycle for reshaping, fit function wrapping, and restoration
3. Delegate to `self.easy_science_multi_fitter.minimizer.sample(...)` for the MCMC
4. Handle multi-model / multi-contrast aggregation as one joint posterior
5. Return structured posterior dict or `PosteriorResults`

The `MultiFitter` currently stores `self.easy_science_multi_fitter` which has `.minimizer` — we'll call `sample()` on it when the minimizer is a `Bumps` instance.

### Step 4 — Create Bayesian analysis module in reflectometry-lib

**New file**: `reflectometry-lib/src/easyreflectometry/analysis/__init__.py`
**New file**: `reflectometry-lib/src/easyreflectometry/analysis/bayesian.py`

The `bayesian.py` module provides:

```python
class PosteriorResults:
    """Container for Bayesian posterior samples with analysis methods."""

    draws: np.ndarray        # (n_samples, n_params)
    param_names: list[str]
    logp: np.ndarray | None
    sampler_state: Any | None

    def summary(self) -> str:
        """Return formatted summary table with mean, sd, HDI for each parameter."""

    def corner(self, **kwargs) -> None:
        """Plot parameter correlation corner plot using the `corner` library."""

    def credible_interval(self, alpha: float = 0.95) -> dict:
        """Return {param_name: (lower, upper)} credible intervals."""

    def gelman_rubin(self) -> dict:
        """Compute R-hat convergence diagnostic."""

def posterior_summary(draws, param_names) -> str: ...
def plot_corner(draws, param_names, **kwargs) -> None: ...
def plot_trace(draws, param_names, **kwargs) -> None: ...
def credible_intervals(draws, param_names, alpha=0.95) -> dict: ...
```

**Posterior predictive functions** (reflectivity & SLD):

```python
def posterior_predictive_reflectivity(
    draws, param_names, model, q_values, n_samples=200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, lower_95, upper_95) reflectivity arrays."""

def posterior_sld_profile(
    draws, param_names, model, n_samples=200
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (z, median, lower_95, upper_95) SLD profile arrays."""
```

Posterior predictive helpers must save original parameter values and errors before applying any posterior draw, and restore them in a `finally` block after prediction. Parameter lookup should use EasyScience parameter `unique_name` values, matching the BUMPS names after removing the minimizer prefix, rather than display names. This avoids leaving the model mutated after plotting and avoids collisions when repeated models or multi-contrast fits contain similarly named parameters.

### Step 5 — Dependencies

Add to `pyproject.toml` in reflectometry-lib:

```toml
[project.optional-dependencies]
bayesian = [
    "corner>=2.2",
    "arviz>=0.18",
]
```

Make `corner` and `arviz` optional imports — the analysis module should work with graceful fallbacks when they're not installed.

### Step 6 — Update exports

**File**: `reflectometry-lib/src/easyreflectometry/__init__.py`

Add `PosteriorResults` and analysis functions to the public API if desired.

### Step 7 — Update the example notebook

**File**: `reflectometry-lib/docs/src/tutorials/advancedfitting/bayesian_bumps.py`

Replace low-level BUMPS calls with the new high-level API:
- `fitter.sample(data, samples=500, burn=100, thin=10)` instead of manual `FitProblem` + `bumps_fit`
- `plot_corner(posterior['draws'], posterior['param_names'])` instead of manual `corner.corner()`
- `posterior_summary(...)` instead of manual numpy statistics

### Step 8 — Tests

**File**: `reflectometry-lib/tests/test_bayesian.py` (new)

```python
def test_sample_basic(): ...
def test_posterior_summary_format(): ...
def test_corner_plot_does_not_crash(): ...
def test_credible_intervals(): ...
def test_sample_seed_reproducibility(): ...
```

Also add a test in `core/tests/` for the `Bumps.sample()` method.

## Architecture Diagram

```
User Code
    │
    ├─ fitter.fit(data)          ──► classical chi² minimization
    ├─ fitter.sample(data, ...)  ──► Bayesian DREAM MCMC
    │
    ▼
reflectometry-lib MultiFitter
    │  ._prepare_fit_arrays()  ← reused from fit()
    │  delegates to ↓
    ▼
easyscience Bumps minimizer
    │  .fit(x, y, weights)          ──► amoeba/newton/lm
    │  .sample(x, y, weights, ...)  ──► DREAM MCMC (NEW)
    │
    ▼
bumps.fitters.fit / FitProblem / Curve
    └──► reflectivity model evaluation via fit_func

Post-hoc analysis:
    reflectometry-lib analysis.bayesian
        ├── PosteriorResults (container)
        ├── plot_corner() → corner.corner()
        ├── posterior_summary() → numpy stats
        └── posterior_predictive_reflectivity() → model.interface.fit_func()
```

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| DREAM may not converge with default settings | Expose `samples`, `burn`, `thin`, `chains`/`population`; document best practices |
| MCMC is 10-100× slower than least-squares | Document that classical fit first is recommended; add progress callback |
| `corner` / `arviz` may not be installed | Make optional dependencies with graceful fallbacks |
| Multi-dataset sampling aggregation | Follow existing `MultiFitter._precompute_reshaping` pattern |
| BUMPS DREAM API changes | Pin bumps version; wrap in our API |
| Reproducibility | Expose `seed` parameter; document how to save/load DreamState |
| Model mutation during posterior prediction | Save/restore original parameter values and map draws by `unique_name` |

## Files to Create / Modify

### Create:
1. `reflectometry-lib/src/easyreflectometry/analysis/__init__.py`
2. `reflectometry-lib/src/easyreflectometry/analysis/bayesian.py`
3. `reflectometry-lib/tests/test_bayesian.py`

### Modify:
4. `core/src/easyscience/fitting/available_minimizers.py` — no `Bumps_dream`; optionally document that samplers are exposed separately
5. `core/src/easyscience/fitting/minimizers/minimizer_bumps.py` — add dedicated `sample()` method without adding `'dream'` to optimizer methods
6. `reflectometry-lib/src/easyreflectometry/fitting.py` — add `sample()` to `MultiFitter`
7. `reflectometry-lib/src/easyreflectometry/__init__.py` — optional: export new classes
8. `reflectometry-lib/pyproject.toml` — add optional `bayesian` dependencies
9. `reflectometry-lib/docs/src/tutorials/advancedfitting/bayesian_bumps.py` — update to use new API

## Implementation Order

1. **core changes** (Steps 1-2): Add `Bumps.sample()` method while keeping DREAM out of optimizer enum/method dispatch
2. **reflectometry-lib fitting** (Step 3): Add `MultiFitter.sample()`
3. **reflectometry-lib analysis** (Step 4): Create `analysis/bayesian.py` with corner plot & stats
4. **Dependencies** (Step 5): Add optional deps to pyproject.toml
5. **Exports** (Step 6): Update `__init__.py`
6. **Example update** (Step 7): Update notebook to use new API
7. **Tests** (Step 8): Add test coverage
