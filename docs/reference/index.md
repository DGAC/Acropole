# Python API

Acropole is a library — it exposes a Python API, not a command-line interface. The public
surface is two classes, both importable from the package root:

```python
from acropole import FuelEstimator, AircraftFuelEstimator
```

## `FuelEstimator`

The high-level, `DataFrame`-in / `DataFrame`-out estimator. Accepts a pandas **or** polars
`DataFrame`, dispatches per aircraft typecode, and returns the same type enriched with
`fuel_flow` (kg/s), `fuel_flow_kgh` (kg/h) and — when a `second` column is given —
`fuel_cumsum` (kg).

- `estimate(flight, **col_mapping)` — score a frame; map column names with keyword
  arguments (see [Map your own columns](../howto/columns-mapping.md)).
- `for_aircraft(typecode)` — return an `AircraftFuelEstimator` bound to a typecode,
  reusing the loaded ONNX session and parameters.

## `AircraftFuelEstimator`

The typecode-bound, **numpy-only** fast path. Built for a single aircraft type with all
per-aircraft constants precomputed; `estimate()` takes 1-D numpy arrays and returns a 1-D
array of fuel flow in kg/s.

---

The full API below — every public class, method and function with its signature and
docstring — is rendered from the source by
[mkdocstrings](https://mkdocstrings.github.io/).

::: acropole.estimator
    options:
      show_root_heading: false
      show_source: true
      members_order: source
      heading_level: 2
