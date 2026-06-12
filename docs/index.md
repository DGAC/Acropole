---
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="assets/logo.png" alt="Acropole Logo" width="180" />
</p>

<p align="center">
  <strong>Acropole — Predict aircraft fuel flow from trajectory data</strong>
</p>

<p align="center">
  <a href="https://github.com/DGAC/Acropole/actions/workflows/ci.yml">
    <img src="https://github.com/DGAC/Acropole/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
  <a href="https://forge.axm-protocols.io/audit/">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DGAC/Acropole/gh-pages/badges/axm-audit.json" alt="axm-audit" />
  </a>
  <a href="https://forge.axm-protocols.io/init/">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DGAC/Acropole/gh-pages/badges/axm-init.json" alt="axm-init" />
  </a>
  <a href="https://github.com/DGAC/Acropole/actions/workflows/ci.yml">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DGAC/Acropole/gh-pages/badges/coverage.json" alt="coverage" />
  </a>
  <a href="https://pypi.org/project/acropole/">
    <img src="https://img.shields.io/pypi/v/acropole" alt="PyPI" />
  </a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+" />
  <a href="https://github.com/DGAC/Acropole/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL-3.0" />
  </a>
</p>

---

`acropole` predicts the **fuel flow of aircraft** (kg/s, kg/h and cumulative kg) from
trajectory data — groundspeed, altitude and vertical rate — using a portable **ONNX**
model trained on Quick Access Recorder (QAR) data. It accepts a **pandas or polars**
`DataFrame`, dispatches per aircraft typecode, and returns the same frame enriched with
fuel-flow columns.

## Features

- ⛽ **Fuel-flow prediction** — `fuel_flow` (kg/s), `fuel_flow_kgh` (kg/h), `fuel_cumsum` (kg)
- ✈️ **Multi-aircraft** — frames mixing typecodes are scored per typecode
- 🐼 **pandas *and* polars** — same type in, same type out; polars engine internally
- 🚀 **Fast ONNX runtime** — 2–4.8× faster than the original TensorFlow model, no TF dependency
- 📈 **Temporal derivatives** — accelerations from a `second` column, or pre-computed
- 🎯 **Column mapping** — point each feature at your own column names
- 💻 **Command-line** — the `acropole estimate` command enriches a CSV/parquet file without writing Python

## Quick Start

```bash
pip install "acropole[pandas]"
```

```python
import pandas as pd
from acropole import FuelEstimator

flight = pd.DataFrame({
    "typecode": ["A320", "A320", "A320", "A320"],
    "groundspeed": [400, 410, 420, 430],
    "altitude": [10000, 11000, 12000, 13000],
    "vertical_rate": [2000, 1500, 1000, 500],
})

flight_fuel = FuelEstimator().estimate(flight)
# adds fuel_flow (kg/s), fuel_flow_kgh (kg/h)
```

Prefer the command line? `acropole estimate flight.csv` writes an enriched
`flight_fuel.csv` — see [Estimate fuel from the command line](howto/cli.md).

## Documentation

This documentation follows the [Diátaxis](https://diataxis.fr/) framework:

- **[Tutorials](tutorials/getting-started.md)** — learn Acropole step by step, from install to your first estimate.
- **[How-To Guides](howto/index.md)** — task-oriented recipes: multi-aircraft frames, column mapping, derivatives.
- **[Reference](reference/index.md)** — the Python API surface, auto-generated from the source.
- **[Explanation](explanation/architecture.md)** — how the polars pipeline, the ONNX model and the typecode dispatch fit together.

---

<div style="text-align: center; margin: 2rem 0;">
  <a href="tutorials/getting-started/" class="md-button md-button--primary">Get Started →</a>
  <a href="reference/" class="md-button">Python API</a>
</div>
