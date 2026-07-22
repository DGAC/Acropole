# Architecture

This page explains how Acropole turns a trajectory `DataFrame` into a fuel-flow estimate:
the backend-agnostic narwhals front end, the numpy compute core, the ONNX model, feature
normalization, and the per-typecode dispatch.

## Overview

```mermaid
graph TD
    IN["DataFrame (pandas, polars, pyarrow, …)"] --> NW["narwhals wrapper (no copy, no conversion)"]
    NW --> COLS["Extract feature columns to numpy, once"]
    COLS --> DISP["Dispatch by typecode (numpy mask)"]
    DISP --> A320["AircraftFuelEstimator(A320)"]
    DISP --> B738["AircraftFuelEstimator(B738)"]
    A320 --> FEAT["Build 12-feature matrix + min/max normalize"]
    B738 --> FEAT
    FEAT --> ONNX["ONNX InferenceSession"]
    ONNX --> SCALE["× FUEL_FLOW_TO × ENGINE_NUM"]
    SCALE --> OUT["fuel_flow, fuel_flow_kgh, fuel_cumsum"]
    OUT --> RET["return same type as input"]
```

## A numpy core behind a backend-agnostic API

`FuelEstimator.estimate()` accepts **any eager `DataFrame` narwhals supports** — pandas,
polars, pyarrow, and others. [narwhals](https://narwhals-dev.github.io/narwhals/) is a
zero-dependency compatibility layer: it wraps the frame you pass in and dispatches to its
native methods, so there is **no conversion to an intermediate frame library**. The input
type is preserved on output — pandas in, pandas out; polars in, polars out.

Consequently `acropole` depends on no frame library at all. Users bring the one they
already have, and the `[polars]`, `[pandas]` and `[pyarrow]` extras are conveniences,
not requirements.

The narwhals layer is deliberately thin: the feature columns are pulled out to numpy
**once**, up front, and everything after that — the per-typecode split, the derivatives,
the feature matrix — is numpy. The frame backend therefore costs one column extraction
per call, not per typecode group, and it never touches the hot path.

The aircraft parameter table is read with the stdlib `csv` module rather than a frame
backend, which is what lets the numpy-only `AircraftFuelEstimator` run with **no
dataframe library installed whatsoever**.

## Dispatch by typecode

A frame may mix several aircraft types. `estimate()` groups rows by `typecode`, and for
each group builds an `AircraftFuelEstimator` bound to that typecode via
`for_aircraft()` — which **reuses the already-loaded ONNX session and the parsed
parameters**, so adding a typecode costs only a parameter lookup, not a model reload.
Each group's predictions are scattered back into the original row order. Rows whose
typecode is unknown stay `NaN` and raise a warning.

`AircraftFuelEstimator` is also the public fast path: bound to one typecode, it works on
plain 1-D numpy arrays with all per-aircraft constants (engine scale, normalization
arrays) precomputed at construction.

## The ONNX model

The estimator runs a single ONNX `InferenceSession` on the CPU. The model takes a
`(N, 12)` matrix of features and emits one fuel-flow value per sample. The twelve
features, in order, are:

```
engine_type, d_altitude, d_groundspeed, d_airspeed, surface,
max_ope_alti, max_ope_speed, altitude, groundspeed, airspeed,
vertical_rate, mass_norm
```

Several of these (`engine_type`, `surface`, `max_ope_alti`, `max_ope_speed`) are
**per-aircraft constants** read from `aircraft_params.csv`; the rest come from the
trajectory (and its derivatives). The raw model output is the per-engine fuel flow; it is
scaled to the whole aircraft by multiplying by `FUEL_FLOW_TO × ENGINE_NUM` (the take-off
reference flow times the engine count), yielding `fuel_flow` in **kg/s**. `fuel_flow_kgh`
is that value × 3600, and `fuel_cumsum` is the running integral over the time step.

## Feature normalization

Before inference every feature is min/max-normalized into `[0, 1]` using fixed bounds —
one `(min, max)` pair per feature, baked into the package. The transform is the plain
`(x − min) / (max − min)`. The bounds encode the physical ranges the model was trained on
(e.g. altitude up to 50 000 ft, vertical rate ±5 000 ft/min). The normalized matrix is
cast to `float32` to match the model's input dtype.

## Why ONNX

Acropole originally ran a TensorFlow model. It was migrated to ONNX for two reasons:

- **No heavy ML framework** — the runtime depends only on `numpy`, `narwhals` and
  `onnxruntime`. There is no TensorFlow install, no GPU toolchain, no multi-hundred-MB
  dependency tree. The model ships as a single portable `.onnx` file inside the package.
- **Speed** — the ONNX path is **2–4.8× faster** than the original TensorFlow inference,
  depending on batch size, with the largest gains on small/medium batches.

The migration was validated to **numerical parity of 1e-6** against the TensorFlow
reference, so existing results are reproduced to floating-point tolerance.

## Design decisions

| Decision | Rationale |
|---|---|
| narwhals front end, numpy core | Works with any frame library; none is a hard dependency, and the hot path stays numpy |
| `narwhals.stable.v1`, not top-level | Backwards-compatibility guarantee across narwhals major versions, as recommended for libraries |
| Same type in / same type out | Drop-in whichever frame library you use, no surprise conversions |
| Params table via stdlib `csv` | A 30-row lookup needs no frame backend, so the numpy-only API has zero of them |
| Per-typecode dispatch with shared session | Process a mixed fleet in one call without reloading the model |
| ONNX over TensorFlow | Portable single-file model, no heavy framework, 2–4.8× faster, 1e-6 parity |
| Fixed min/max normalization | Deterministic, baked-in physical ranges matching the training distribution |
| `src/` layout + `py.typed` | PEP 561 typed package, no import-shadowing, strict mypy |
