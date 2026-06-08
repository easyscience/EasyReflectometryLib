# Design: Polarized ORSO experiment file reader

Status: Draft for review Scope: Extend the ORSO `.ort` reader to
recognize polarized datasets and, where possible, assign a spin
direction to each dataset. Initial implementation focuses on
**unpolarized (`un`)** and the **half-polarized (`po`/`mo`)
spin-up/spin-down** cases. All other polarization states are recognized
but routed to the existing multi-experiment load path with a warning.

---

## 1. Background

### 1.1 Current reader

ORSO loading lives in
[`src/easyreflectometry/orso_utils.py`](src/easyreflectometry/orso_utils.py):

- `LoadOrso(orso_data)` → `(sample, data)`; entry point used by
  `Project.load_orso_file`
  ([`project.py:373`](src/easyreflectometry/project.py#L373)).
- `load_orso_data(orso_data)` → `sc.DataGroup` with `data`, `coords`,
  `attrs`. Iterates every dataset `o` in the parsed list and names
  entries `R_<name>` / `Qz_<name>`, where `name` is `o.info.data_set`
  (falling back to the numeric index). The full ORSO header is stashed
  in `attrs[f'R_{name}']['orso_header']`.
- `load_data_from_orso_file(fname)` → loads via `orso.load_orso` then
  `load_orso_data`.
- The thin wrapper in
  [`data/measurement.py`](src/easyreflectometry/data/measurement.py)
  (`load`, `load_as_dataset`, `merge_datagroups`) consumes the
  DataGroup.

The reader is **already multi-dataset aware** — a multi-angle file
produces several `R_*`/`Qz_*` entries in one DataGroup. It is **not**
polarization aware: it ignores `measurement.polarization` and does
nothing special with spin.

### 1.2 Calculator state

The refl1d wrapper
([`calculators/refl1d/wrapper.py`](src/easyreflectometry/calculators/refl1d/wrapper.py))
has `ALL_POLARIZATIONS = False` and raises `NotImplementedError` for
full polarized reflectivity. **This design does not change the
calculators.** It only makes the _reader_ able to identify and label
polarized data so the rest of the stack can grow into it. Spin-resolved
fitting is out of scope here.

### 1.3 The ORSO metadata

Per ORSO 1.1, polarization is recorded **per dataset** at:

```
o.info.data_source.measurement.instrument_settings.polarization
```

The spin of a follow-up dataset is recorded at the dataset level:

```
o.info.data_set            # e.g. "spin-up", "spin_down"
```

The ORSO-allowed polarization states are:

```
un  po  mo  op  om  pp  pm  mp  mm  vector
```

(`un` = unpolarized; first char = incident, second = analyzed; `o` =
open/none.)

#### Real-file reality (from `tests/_static/`)

Inspecting the sample files shows the metadata is **messier than the
spec**:

| File                             | n   | `polarization` (parsed) | `data_set` labels                    |
| -------------------------------- | --- | ----------------------- | ------------------------------------ |
| `NOB_reflectivity_polarized.ort` | 2   | `po`, `mo` (enum)       | `spin-up`, `spin-down`               |
| `test_example2.ort`              | 2   | `p`, `p` (raw str)      | `spin_up`, `spin_down`               |
| `test_example3.ort`              | 3   | `p`, `p`, `p`           | `spin_up`, `spin_down`, `spin_three` |
| `test_example4.ort`              | 2   | `p`, `p`                | `spin_up`, `spin_down`               |
| `example.ort`                    | 1   | `unpolarized` (enum)    | `None`                               |

Consequences the implementation must absorb:

1. **The value may be an `orsopy` enum or a raw string.** When the value
   is a valid member, orsopy gives `Polarization.po` (use `.value` →
   `"po"`). When it is _not_ a valid member (e.g. `p`), orsopy emits an
   `ORSOSchemaWarning` and stores the **raw string** `"p"`.
   Normalization must handle both.
2. **Spelling varies**: `unpolarized` (full word) vs the `un` code;
   single-letter `p`/`m` vs two-letter `po`/`mo`.
3. **Spin labels vary**: hyphen (`spin-up`) vs underscore (`spin_up`).
4. **Not every follow-up label is a spin direction**: `spin_three` is
   unrecognizable → this is exactly the "cannot determine unequivocally"
   branch.

---

## 2. Goals / non-goals

**Goals**

- Detect whether an `.ort` file is polarized from
  `measurement.polarization`, tolerant of enum-vs-string and spelling
  variants.
- For the half-polarized `po`/`mo` (and single-letter `p`/`m`) case,
  assign a spin direction (`up` / `down`) to each dataset:
  - primary source: `data_set` string;
  - fallback: the per-dataset polarization code (`p*`→up, `m*`→down).
- When **all** datasets in a polarized file get an unambiguous spin,
  expose a **polarized container** that marks the experiment as
  polarized and carries the per-spin sub-datasets.
- When spin **cannot** be resolved for every dataset (e.g. `spin_three`,
  missing labels, mixed/unsupported states), fall back to the **existing
  multi-experiment DataGroup load** and emit a `UserWarning`.
- Leave the unpolarized path behavior unchanged.

**Non-goals**

- No polarized reflectivity calculation / fitting (calculators
  untouched).
- No support yet for fully-analyzed (`pp/pm/mp/mm`), `op/om`, or
  `vector` — recognized only insofar as routing them to fallback +
  warning.
- No `.ort` _writing_ changes.

---

## 3. Polarization taxonomy & normalization

A single helper normalizes whatever orsopy produced into a small
vocabulary.

### 3.0 Robust metadata access

Before normalizing, the value must be fetched defensively. `example.ort`
has an `instrument_settings` section with **no** `polarization` key;
other files may omit `measurement` entirely. Every level of the path is
therefore optional:

```python
def _get_polarization(o):
    """Per-dataset polarization value as a raw str, or None if absent."""
    try:
        raw = o.info.data_source.measurement.instrument_settings.polarization
    except AttributeError:
        return None
    if raw is None:
        return None
    return raw.value if hasattr(raw, 'value') else str(raw)   # enum vs raw string
```

The same defensiveness applies to `o.info.data_set` (may be `None`).
(Addresses review #2 and #10.)

### 3.1 Normalize the polarization value

```
_normalize_polarization(raw) -> str | None
```

- Accept an enum (read `.value`), a string, or `None`.
- Lowercase, strip.
- The allowed vocabulary is exactly the ORSO list — no invented codes:
  ```
  un po mo op om pp pm mp mm vector
  ```
- Map to a canonical code:
  - starts with `un` (`unpolarized`, `un`) → `"un"`
  - exactly `vector` → `"vector"`
  - one of `po mo op om pp pm mp mm` → itself
  - single char `p` or `m` → **legacy polarized-presence hint only**,
    canonicalize to `"p"` / `"m"`. These mark the file as polarized but
    carry **no** spin meaning (review #2: fixtures use `p` for _both_
    spin-up and spin-down datasets). They are never expanded to
    `po`/`mo`.
  - single char `o`, or anything else, or `None` → `None`
    (unknown/unsupported). We do **not** invent `oo` (review #7 — `oo`
    is not an allowed state).

### 3.2 Classify a file

```
_classify_polarization(orso_data) -> Literal["unpolarized", "half_polarized", "unsupported"]
```

Rules, evaluated over the per-dataset normalized codes:

- All datasets `un` (or polarization metadata absent everywhere) →
  `unpolarized`.
- Every dataset code in `{po, mo, p, m}` (i.e. incident-only or legacy
  presence hint) → `half_polarized`. **This is the in-scope polarized
  case.** Note that `p`/`m` mark the file polarized but do not
  themselves resolve spin (§4).
- Anything else present (`op om pp pm mp mm vector`, or a mix that
  includes an unsupported state) → `unsupported`.

> Decision: classification is driven by the **per-dataset** codes,
> because the sample files prove the value lives on each dataset and can
> differ across them (`po` + `mo`). A file is "half-polarized" iff _all_
> its datasets are incident-only polarized.

### 3.3 Documented edge cases (review #3)

- **Mixed `un` + half-polarized in one file.** Not all datasets share a
  class. Decision: classify as `unsupported` → fallback DataGroup +
  warning. We do not attempt to mark only some datasets polarized in the
  first iteration.
- **Single-dataset half-polarized** (e.g. `test_example1.ort`: one
  dataset, `p`, `spin_up`). Decision: classify as `half_polarized` and
  produce a `PolarizedData` with a **single** spin channel, but emit a
  `UserWarning` that only one spin direction is present (no companion
  channel). A consumer can still read the one resolved spin; the warning
  prevents silent misinterpretation.
- **Contradiction** (`polarization: un` but `data_set` looks like a
  spin, or vice-versa). Decision: the polarization classification wins;
  if it says `unpolarized` we do not invent spin channels. No warning
  (spin-like labels are legal dataset names in unpolarized multi-angle
  files).

---

## 4. Spin assignment

For a `half_polarized` file, assign each dataset a spin ∈ {`up`,
`down`}.

### 4.1 From `data_set` (primary)

```
_spin_from_data_set(label) -> "up" | "down" | None
```

- Normalize: lowercase, replace `-`/`_`/whitespace with a single space,
  strip a leading `spin ` prefix.
- Map:
  - `{up, u, +, plus, spin up}` → `up`
  - `{down, d, -, minus, spin down}` → `down`
  - anything else (e.g. `three`) → `None`

### 4.2 `data_set` is the sole spin source (revised per review #1)

**Decision (supersedes the earlier Q2 "polarization fallback"):** spin
is determined **only** from the `data_set` label. There is no
polarization-code fallback. A resolved polarized load requires a
recognized `data_set` spin label on _every_ dataset. This matches the
original requirement ("the follow-up data should have a `data_set`
metadata describing the spin direction… if we can determine all datasets
with the metadata, this is unequivocal; otherwise warn").

The polarization code is used for two things only: (a) classifying the
file as polarized (§3.2), and (b) a **consistency cross-check** (§4.4) —
never to assign a spin.

```
spin = _spin_from_data_set(o.info.data_set)      # None ⇒ not resolvable ⇒ fallback
```

### 4.3 Polarization-code → expected spin (cross-check only)

```
_spin_from_polarization_code(code) -> "up" | "down" | None
```

- `po`, `pp`, `pm` → `up` (incident `+`)
- `mo`, `mp`, `mm` → `down` (incident `-`)
- `p`, `m`, anything else → `None` (no spin meaning; presence hint only)

Used solely in §4.4 to detect contradictions. Never used to fill a
missing spin.

### 4.4 Unequivocal vs. fallback

The assignment is **unequivocal** (→ `PolarizedData`) iff _all_ hold:

1. every dataset has a recognized `data_set` spin (non-`None` from
   §4.1);
2. **no contradiction** with the polarization code: when
   `_spin_from_polarization_code(code)` is non-`None`, it must equal the
   `data_set` spin (review #3 — e.g. `polarization: po` +
   `data_set: spin_down` is contradictory). A `None` code (incl.
   `p`/`m`) imposes no constraint;
3. dataset identity is unique (§4.5).

If any condition fails → **fallback**: load all datasets via the
standard multi-experiment path (§6/§7.2) and emit a `UserWarning` naming
the reason (missing/unrecognized label, contradiction, or duplicate
identity).

Worked examples:

- `NOB` (`po`+`spin-up`, `mo`+`spin-down`): resolved; code agrees with
  labels.
- `test_example2`/`4` (`p`+`spin_up`, `p`+`spin_down`): resolved from
  labels; `p` imposes no constraint.
- `test_example3` (`spin_three`): condition 1 fails → fallback.
- Synthetic `po`+`spin_down`: condition 2 fails → fallback.

### 4.5 Identity & collision rule (revised per review #4)

Two datasets sharing the same **spin** is _not_ an error (e.g.
multi-angle spin-up × 2). The requirement is unique **identity**:

- Multiple datasets may share a spin **iff** each has a unique,
  recognized `data_set` label.
- **Duplicate or missing `data_set` labels are ambiguous** for the
  container and trigger fallback — unless a unique key can be formed
  without discarding the original label.

Container key scheme: `"<data_set>"` (the recognized label) is the
natural unique key; the spin is stored alongside in metadata. If two
datasets carry the _same_ `data_set` string, identity is ambiguous →
fallback. The original `data_set` and full ORSO header always remain in
`attrs` (see §4.6).

`test_example3` (`spin_up`, `spin_down`, `spin_three`) → `spin_three`
resolves to `None` → **fallback** (load all 3 as a normal
multi-experiment DataGroup + warn).

### 4.6 Duplicate-key preservation in `load_orso_data` — in scope (review #5)

`load_orso_data` keys entries by `o.info.data_set` and **overwrites** on
duplicate keys (`data[f'R_{name}'] = ...`). Multi-dataset files with
repeated `data_set` values silently lose datasets today.

**This is a prerequisite, not adjacent.** The fallback requirement is
"load _all_ sets." But the standard fallback path is
`Project.load_all_experiments_from_file`
([project.py:574](src/easyreflectometry/project.py#L574)), which
iterates `sorted(data_group['data'].keys())` produced by
`load_orso_data`. If `load_orso_data` collapses duplicate keys, the
fallback loads **fewer** experiments than the file contains — directly
violating the requirement.

Therefore `load_orso_data` will disambiguate repeated keys as part of
this work:

- first use: `R_<name>` / `Qz_<name>`
- repeats: `R_<name>_1` / `Qz_<name>_1`, then `_2`, …

The original ORSO header and original `data_set` value remain untouched
in `attrs` so nothing is lost. This lands as a small, separately-tested
change.

---

## 5. Output representation

Per the agreed decision, introduce a **dedicated polarized container**
rather than overloading dataset names.

### 5.1 `PolarizedData` container

A light wrapper (dataclass-style) returned only for the resolved
half-polarized case:

```python
@dataclass
class PolarizedData:
    polarization: str                 # canonical, e.g. "half_polarized"
    spin_channels: dict[str, sc.DataGroup]   # key -> single-dataset DataGroup
    spin_by_key: dict[str, str]       # key -> "up" | "down"
    raw: sc.DataGroup                 # the full merged DataGroup (all channels)
```

- `spin_channels[key]` reuses the **existing** `load_orso_data`
  machinery per dataset, so column/coord/attr handling is unchanged.
- `raw` is the merged DataGroup (same shape today's code produces), so
  any consumer that only understands the flat structure keeps working
  via `.raw`.
- The per-dataset `attrs[...]['orso_header']` already contains the full
  ORSO header (including `polarization` and `data_set`); we additionally
  write a small `attrs[...]['spin']` scalar (`"up"`/`"down"`) for direct
  access.

> Rationale: a typed container makes the polarized intent explicit and
> gives the UI / future calculator a stable place to read spin channels,
> while `.raw` preserves the existing contract. This avoids breaking
> `load`/`load_as_dataset` consumers that expect a `sc.DataGroup`.

### 5.2 Backward-compatible accessors (revised per review #6)

**`LoadOrso` and `load_data_from_orso_file` keep their current return
types.** `LoadOrso` always returns `(Sample, sc.DataGroup)` and
`load_data_from_orso_file` always returns `sc.DataGroup`. This is a hard
constraint: `test_orso_utils.py` and `Project.load_orso_file` both rely
on it, and the latter assigns attributes (`.name`, `.model`) onto the
returned object. My earlier draft let `LoadOrso` return `PolarizedData`
— that is **withdrawn**.

The polarized richness is exposed only through a **new** entry point:

```
load_polarized_orso_data(orso_data) -> PolarizedData | sc.DataGroup
```

It returns `PolarizedData` only for the resolved half-polarized case and
a plain `sc.DataGroup` otherwise (unpolarized / unsupported / unresolved
fallback).

So the spin information is available **two** ways, and both are
populated:

1. **In-band**, on the flat DataGroup that the legacy path already
   returns: each dataset's `attrs[f'R_{name}']['spin']` scalar carries
   `"up"`/`"down"` when resolved, and the DataGroup gets a top-level
   `attrs['polarization']` marker. Legacy consumers ignore it;
   spin-aware consumers can read it without the typed container.
2. **Typed**, via `load_polarized_orso_data` → `PolarizedData`.

This keeps the agreed "dedicated polarized container" (Q3) while
guaranteeing nothing downstream breaks.

---

## 6. Control flow

```
load (orso parsed list)
        │
        ▼
_classify_polarization
        │
 ┌──────┼─────────────────────────────┐
 │      │                             │
unpolarized   half_polarized      unsupported
 │      │                             │
 │      ▼                             │
 │   assign spins (§4)                │
 │      │                             │
 │  unequivocal? ── no ──────────────►│
 │      │ yes                         │
 │      ▼                             ▼
 │  PolarizedData            warn(UserWarning) +
 │  (spin channels)          standard multi-experiment
 │                           DataGroup (today's behavior)
 ▼
standard DataGroup (today's behavior, unchanged)
```

Warnings (all `UserWarning`, `stacklevel` tuned):

- Unsupported state(s): _"ORSO polarization state '<code>' is not yet
  supported; loading all datasets without spin assignment."_
- Half-polarized but spin unresolved (missing/unrecognized `data_set`):
  _"Could not determine the spin direction for all datasets
  (unrecognized data_set label '<x>'); loading as a standard
  multi-dataset experiment."_
- Contradiction (`data_set` vs polarization code): _"Polarization code
  '<code>' contradicts data_set spin '<spin>'; loading as a standard
  multi-dataset experiment."_
- Single-channel half-polarized: _"Only one spin direction ('<spin>')
  present; no companion channel."_

The existing orsopy `ORSOSchemaWarning` for raw values like `p` is
independent and will still appear; we do not suppress it.

> **Fallback = the existing multi-experiment workflow (review #6).**
> "Standard multi-experiment load" is not an abstraction — at the
> `Project` level it means the unresolved/unsupported case routes
> through `Project.load_all_experiments_from_file`
> ([project.py:574](src/easyreflectometry/project.py#L574)), which
> registers each dataset as an independent experiment. The reader still
> returns a flat `sc.DataGroup`; §7.2 specifies how the project chooses
> that route.

---

## 7. API surface & placement

Per the agreed "extend `orso_utils`, update project loader" decision.

### 7.1 `orso_utils.py` (new/changed)

- `_normalize_polarization(raw) -> str | None` _(new, private)_
- `_classify_polarization(orso_data) -> str` _(new, private)_
- `_spin_from_data_set(label) -> str | None` _(new, private; sole spin
  source)_
- `_spin_from_polarization_code(code) -> str | None` _(new, private;
  **cross-check only**, not a spin source — §4.3, review #1/#2)_
- `_resolve_spins(orso_data) -> list[str] | None` _(new, private; `None`
  ⇒ fallback)_
- `load_polarized_orso_data(orso_data) -> PolarizedData | sc.DataGroup`
  _(new, public)_
- `load_orso_data(...)` — reused per channel and for fallback; gains
  in-band `attrs['spin']` / top-level `attrs['polarization']` tagging
  and **duplicate-key disambiguation** (review #5, now in scope — §4.6).
- `LoadOrso(orso_data)` — **return type unchanged**: still
  `(Sample, sc.DataGroup)`. It does _not_ return `PolarizedData` (review
  #6).

`PolarizedData` lives in `orso_utils.py` (or `data/` if reused widely).

### 7.2 `Project.load_orso_file` ([`project.py:373`](src/easyreflectometry/project.py#L373))

`_experiments` is a `dict` keyed by integer, and the loader assigns
attributes onto the stored value (`self._experiments[0].name = ...`,
`self._experiments[0].model = ...`). A `PolarizedData` is **not** a
`DataGroup` and would break those lines (review #5). Therefore:

- **Resolved-polarized**: store the flat `sc.DataGroup` (in-band
  spin/polarization `attrs`) — never a `PolarizedData` — and
  additionally call `load_polarized_orso_data`, stashing the
  `PolarizedData` on a separate nullable attribute (e.g.
  `self._experiment_polarization`) plus a UI marker.
- **Unresolved / unsupported (fallback)**: route to the **existing
  multi-experiment workflow** —
  `self.load_all_experiments_from_file(path)`
  ([project.py:574](src/easyreflectometry/project.py#L574)) — so _every_
  dataset is registered as an independent experiment (review #6). This
  is why the duplicate-key fix (§4.6) is a prerequisite: that method
  keys off `data_group['data'].keys()`.
- **Unpolarized**: behavior unchanged.

Exact experiment-model plumbing (one experiment with channels vs. one
experiment per spin) is a follow-up decision flagged in §9; the reader
contract above does not depend on it.

### 7.3 `data/measurement.py`

- `load(...)` stays returning `sc.DataGroup` (uses fallback/`.raw`).
- Optionally add
  `load_polarized(fname) -> PolarizedData | sc.DataGroup`.
- `merge_datagroups` unchanged (used to build `.raw`).

---

## 8. Test plan

Use existing fixtures in `tests/_static/`:

| Case                                    | File                                            | Expected                                                                                                   |
| --------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Unpolarized single                      | `example.ort` / `Ni_example.ort`                | classify `unpolarized`; `sc.DataGroup`; unchanged output                                                   |
| Half-pol, enum `po`/`mo`, hyphen labels | `NOB_reflectivity_polarized.ort`                | classify `half_polarized`; spins `up`,`down`; `PolarizedData`                                              |
| Half-pol, raw `p`, underscore labels    | `test_example2.ort`, `test_example4.ort`        | spins from `data_set`; `PolarizedData`                                                                     |
| Half-pol, unresolved 3rd label          | `test_example3.ort`                             | fallback DataGroup + `UserWarning`                                                                         |
| Unsupported state                       | synthetic `pp`/`mm`, `op`, `om`, `mp`, `vector` | fallback DataGroup + `UserWarning`, **all datasets preserved**                                             |
| Single-channel half-pol                 | `test_example1.ort`                             | `PolarizedData` with one `up` channel (resolved via `data_set`) + `UserWarning` (review #3)                |
| Per-dataset override regression         | `NOB_reflectivity_polarized.ort`                | assert `_get_polarization` gives `po`, `mo` for datasets 0/1 (review #1)                                   |
| Backward-compat contract                | any                                             | `LoadOrso` returns `(Sample, sc.DataGroup)`; `load_data_from_orso_file` returns `sc.DataGroup` (review #6) |
| **No data_set → fallback**              | synthetic: `po`/`mo`, `data_set=None`           | fallback + `UserWarning` (review #1 — no code fallback)                                                    |
| **Raw `p` not spin-up**                 | synthetic: `p`/`p`, `data_set=None`             | fallback; `p` never becomes `up` (review #2)                                                               |
| **Contradiction**                       | synthetic: `po`+`spin_down`, `mo`+`spin_up`     | fallback + `UserWarning` (review #3)                                                                       |
| **Duplicate `data_set` preserved**      | synthetic: two datasets, same label             | fallback registers **both** experiments, none overwritten (review #5)                                      |
| **Mixed un + polarized**                | synthetic: `un` + `po`                          | classify `unsupported` → fallback + `UserWarning` (review #3/§3.3)                                         |

Unit tests for `_get_polarization` (missing `instrument_settings`,
missing `polarization`, enum, raw string), `_normalize_polarization`
(enum, `unpolarized`, `p`, `po`, junk, `None`) and `_spin_from_data_set`
(`spin-up`, `spin_down`, `up`, `+`, `three`).

> Test-harness note (review #11): files with raw `p` codes emit orsopy's
> `ORSOSchemaWarning` at parse time. Tests asserting on _our_ >
> `UserWarning` must filter/ignore that orsopy warning (or use
> `pytest.warns(UserWarning, match=...)` which checks for a matching
> warning rather than the only warning).

---

## 9. Open questions / follow-ups (not blocking this reader)

1. **Experiment plumbing**: should a resolved polarized file become one
   experiment holding N spin channels, or N experiments (one per spin)?
   Affects `Project` and the UI, not the reader contract.
2. **Calculator integration**: spin-resolved simulation/fitting is gated
   on refl1d's `ALL_POLARIZATIONS` work (currently
   `NotImplementedError`).
3. **`PolarizedData` home**: keep in `orso_utils` or promote to `data/`
   if other importers need it.
4. **Extending scope** to `op/om`, `pp/pm/mp/mm`, `vector` later: the
   classifier and spin maps are structured to grow; only the
   `unsupported` branch and the spin-code map need extension.

---

## 10. Disposition of review POLARIZED_ORSO_REV1.md

| #   | Issue                                                            | Disposition                                                                     | Where      |
| --- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------- |
| 1   | Verify per-dataset polarization merge                            | **Adopted** — verified (`po`/`mo`); added regression test                       | §8         |
| 2   | Polarization access-path fragility                               | **Adopted** — `_get_polarization` getattr chain                                 | §3.0       |
| 3   | Classification edge cases (single-channel, mixed, contradiction) | **Adopted** — documented behavior                                               | §3.3       |
| 4   | `data_set` key collision in `load_orso_data`                     | **Adopted** — later upgraded to in-scope prerequisite by REV2 #5                | §4.6       |
| 5   | Don't store `PolarizedData` in `_experiments`                    | **Adopted** — store `.raw` DataGroup; container on a separate attr              | §7.2       |
| 6   | `LoadOrso` return type must not change                           | **Adopted** — withdrew earlier draft; `LoadOrso` stays `(Sample, sc.DataGroup)` | §5.2, §7.1 |
| 7   | NOB 5-column file                                                | **Rejected** — review self-corrected; column indexing works                     | —          |
| 8   | Same-polarization fallback collapse                              | **Adopted (note)** — already covered by §4.5; clarified as intended             | §4.5–4.6   |
| 9   | Shared vs per-dataset header merge                               | **Rejected** — orsopy merges correctly; no action                               | —          |
| 10  | Missing `instrument_settings` section                            | **Adopted** — folded into `_get_polarization`                                   | §3.0       |
| 11  | `ORSOSchemaWarning` in tests                                     | **Adopted (note)** — test-harness guidance                                      | §8         |

## 11. Disposition of review POLARIZED_ORSO_REV2.md

| #   | Issue                                                                | Disposition                                                                             | Where      |
| --- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------- |
| 1   | Make "unequivocal" stricter — require `data_set`, drop code fallback | **Adopted** (user-confirmed, overrides earlier Q2) — `data_set` is the sole spin source | §4.2, §4.4 |
| 2   | Do not treat raw `p` as spin-up                                      | **Adopted** — `p`/`m` are presence hints only, never spin; fixes a real mislabel bug    | §3.1, §4.3 |
| 3   | Explicit conflict handling (`po` + `spin_down`)                      | **Adopted** — contradiction → fallback + warning                                        | §4.4       |
| 4   | Precise duplicate/collision rule                                     | **Adopted** — unique _identity_ required; dup spin OK, dup/missing label → fallback     | §4.5       |
| 5   | Duplicate-key preservation is in-scope                               | **Adopted** — upgraded from adjacent to prerequisite (fallback must load all sets)      | §4.6       |
| 6   | Define project-level fallback explicitly                             | **Adopted** — routes to `Project.load_all_experiments_from_file`                        | §6, §7.2   |
| 7   | Don't canonicalize single `o` to `oo`                                | **Adopted** — `o` treated as unknown/unsupported                                        | §3.1       |
