<p align="center">
  <img src="https://raw.githubusercontent.com/DGAC/Acropole/main/logo.png" alt="Acropole Logo" width="180" />
</p>

<p align="center">
  <strong>Acropole — Predict aircraft fuel flow from trajectory data</strong>
</p>

<p align="center">
  <a href="https://github.com/DGAC/Acropole/actions/workflows/ci.yml"><img src="https://github.com/DGAC/Acropole/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://DGAC.github.io/Acropole/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DGAC/Acropole/gh-pages/badges/axm-audit.json" alt="axm-audit"></a>
  <a href="https://DGAC.github.io/Acropole/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DGAC/Acropole/gh-pages/badges/axm-init.json" alt="axm-init"></a>
  <a href="https://DGAC.github.io/Acropole/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DGAC/Acropole/gh-pages/badges/coverage.json" alt="coverage"></a>
  <a href="https://pypi.org/project/acropole/"><img src="https://img.shields.io/pypi/v/acropole" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <a href="https://github.com/DGAC/Acropole/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL-3.0"></a>
  <a href="https://DGAC.github.io/Acropole/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
</p>

---

`acropole` predicts the **fuel flow of aircraft** (kg/s, kg/h and cumulative kg) from
trajectory data — groundspeed, altitude and vertical rate — using a portable **ONNX**
model trained on Quick Access Recorder (QAR) data. It accepts a **pandas or polars**
`DataFrame`, dispatches per aircraft typecode, and returns the same frame enriched with
fuel-flow columns. The runtime depends only on numpy + polars + onnxruntime — no heavy
ML framework.

📖 **[Full documentation](https://DGAC.github.io/Acropole/)**

## Features

- ⛽ **Fuel-flow prediction** — per-sample `fuel_flow` (kg/s), `fuel_flow_kgh` (kg/h) and `fuel_cumsum` (kg) from a flight trajectory
- ✈️ **Multi-aircraft** — frames mixing several typecodes are scored per typecode, each row with its own aircraft parameters
- 🐼 **pandas *and* polars** — `estimate()` accepts either and returns the same type; the engine runs on polars internally
- 🚀 **Fast ONNX runtime** — migrated from TensorFlow, **2–4.8× faster** depending on batch size, numerical parity validated to **1e-6**; no TensorFlow dependency
- 📈 **Temporal derivatives** — supply a `second` column to compute accelerations (and `fuel_cumsum`), or pass pre-computed derivatives directly
- 🎯 **Column mapping** — map your own column names with keyword arguments, no renaming required
- ⚡ **Typecode-bound fast path** — `AircraftFuelEstimator` offers numpy-only I/O with pre-computed parameters for hot loops
- 🧪 **Fully typed** — ships `py.typed`, strict mypy on source *and* tests, 98% coverage

## Installation

`acropole` is published on PyPI:

```bash
uv add acropole      # or: pip install acropole
```

The core library runs on numpy + polars + onnxruntime. To pass and receive pandas
`DataFrame`s, install the optional `pandas` extra:

```bash
uv add "acropole[pandas]"      # or: pip install "acropole[pandas]"
```

To work from source, this project uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DGAC/Acropole.git
cd Acropole
uv sync
```

## Quick Start

```python
import pandas as pd
from acropole import FuelEstimator

fe = FuelEstimator()

flight = pd.DataFrame({
    "typecode": ["A320", "A320", "A320", "A320"],
    "groundspeed": [400, 410, 420, 430],     # kt
    "altitude": [10000, 11000, 12000, 13000],  # ft
    "vertical_rate": [2000, 1500, 1000, 500],   # ft/min
    # optional features:
    "second": [0.0, 4.0, 8.0, 12.0],
    "airspeed": [400, 410, 420, 430],
    "mass": [60000, 60000, 60000, 60000],
})

flight_fuel = fe.estimate(flight)
# adds fuel_flow (kg/s), fuel_flow_kgh (kg/h), fuel_cumsum (kg)
```

Notes:

- Providing the `second` column makes the estimation more accurate (it derives
  **accelerations** of the speeds) and enables `fuel_cumsum`.
- `airspeed` is optional; if absent it defaults to `groundspeed`. Accurate airspeed
  improves the estimate.
- The expected sampling rate is **4 seconds**. A higher or lower rate produces noisier
  fuel flow — resample before estimating.

For a full notebook, see [`examples/fuel_estimation.ipynb`](examples/fuel_estimation.ipynb).

## Advanced Usage

### Mapping your own column names

`estimate()` accepts keyword arguments mapping each logical feature to the column
name in your frame. The example flight (QAR columns) maps like this:

```python
import pandas as pd
from acropole import FuelEstimator

flight = pd.read_csv("examples/example_flight.csv")
flight = flight.iloc[::4]  # resample to ~4 s

fe = FuelEstimator()
flight_fuel = fe.estimate(
    flight,
    typecode="FLPL_AIRC_TYPE",
    groundspeed="GRND_SPD_KT",
    altitude="ALTI_STD_FT",
    vertical_rate="VERT_SPD_FTMN",
    second="FLIGHT_TIME",
    airspeed="TRUE_AIR_SPD_KT",
    mass="MASS_KG",
)
```

Required logical columns: `typecode`, `groundspeed` (kt), `altitude` (ft),
`vertical_rate` (ft/min). Optional: `second`, `airspeed`, `mass`, and the
pre-computed derivatives `d_altitude` / `d_groundspeed` / `d_airspeed`.

### Multi-aircraft frames

A single frame can mix typecodes; each row is scored with its own aircraft
parameters. Unsupported typecodes emit a warning and leave `fuel_flow` as `NaN`:

```python
flight = pd.DataFrame({
    "typecode": ["A320", "A320", "B738", "B738"],
    "groundspeed": [400, 410, 420, 430],
    "altitude": [10000, 11000, 12000, 13000],
    "vertical_rate": [2000, 1500, 1000, 500],
})
fe.estimate(flight)  # A320 rows and B738 rows each use their own params
```

### Typecode-bound fast path

For hot loops on a single aircraft, `AircraftFuelEstimator` works directly on numpy
arrays (no per-call params lookup) and returns fuel flow in **kg/s**:

```python
import numpy as np
from acropole import FuelEstimator

# share the already-loaded ONNX session and parameters
afe = FuelEstimator().for_aircraft("A320")

fuel_flow = afe.estimate(
    groundspeed=np.array([400.0, 410.0, 420.0]),
    altitude=np.array([10000.0, 11000.0, 12000.0]),
    vertical_rate=np.array([2000.0, 1500.0, 1000.0]),
)  # kg/s  (multiply by 3600 for kg/h)
```

### Custom aircraft parameters and model

```python
fe = FuelEstimator(
    aircraft_params_path="path/to/your/aircraft_params.csv",
    model_path="path/to/your/model.onnx",
)
```

Aircraft parameters from open data are bundled in `src/acropole/data/aircraft_params.csv`
and the ONNX model in `src/acropole/models/`; both are loaded by default.

## Command-Line

Installing `acropole` also installs the `acropole` command (entry point
`acropole.cli:main`). It reads a flight from a CSV or parquet file, estimates fuel flow,
and **writes the enriched table back to disk** — adding `fuel_flow` (kg/s),
`fuel_flow_kgh` (kg/h) and, with `--second`, `fuel_cumsum` (kg):

```bash
acropole estimate examples/example_flight.csv \
  --typecode FLPL_AIRC_TYPE --groundspeed GRND_SPD_KT --altitude ALTI_STD_FT \
  --vertical-rate VERT_SPD_FTMN --airspeed TRUE_AIR_SPD_KT --mass MASS_KG \
  --second FLIGHT_TIME --out result.csv
# wrote 7796 rows with fuel columns to result.csv
```

If your file already uses the standard column names, no mapping is needed and the output
defaults to `<flight>_fuel.<ext>` next to the input:

```bash
acropole estimate flight.csv        # writes flight_fuel.csv
```

Each `--typecode`, `--groundspeed`, `--altitude`, `--vertical-rate`, `--airspeed`,
`--mass` and `--second` flag maps a logical feature to the matching column in your file
(defaults are the standard names). See the
[CLI reference](https://DGAC.github.io/Acropole/reference/cli/) for all options.

## Comparison of Different Model Performances

Comparison of different model performances per phase for 1000 test flights of A320-214 aircraft using real mass and true airspeed.

| Phase | Samples \# | Metric | ACROPOLE | OpenAP | OpenAP V2 | BADA  | Poll-Schumann |
|-------|------------|--------|--------------|--------------|------------|---------------|----------------|
|       |            | **MAPE (%)**  | 2.13    | 30.35                     | 8.84       | 6.53                       | 6.85                                       |
| CLIMB | 1,403,850   | **MAE (kg/min)** | 1.66          | 25.81                     | 6.92       | 5.53                       | 5.65                                       |
|       |            | **ME (kg/min)**  | 0.85     | -25.66                    | -2.48      | -5.27                      | -4.62                                      |
||||||||||
|       |            | **MAPE (%)**  | 4.41   | 18.59                     | 10.69      | 7.01                       | 4.84                                       |
| LEVEL | 4,017,801   | **MAE (kg/min)** | 1.82      | 7.82                      | 3.48       | 2.65                       | 2.03                                       |
|       |            | **ME (kg/min)**  | 1.22     | -7.47                     | 2.64       | -1.43                      | -0.73                                      |
||||||||||
|       |            | **MAPE (%)**  | 12.63      | 51.69                     | 32.4       | 21.50                      | 21.55                                      |
| DESCENT| 1,684,117  | **MAE (kg/min)** | 2.71         | 8.62                      | 5.58       | 3.71                       | 4.71                                       |
|       |            | **ME (kg/min)**  | 1.88         | -1.75                     | -1.16      | -0.64                      | -3.67                                      |
||||||||||
|       |            | **MAPE (%)**  | 5.91       | 27.60                     | 14.71      | 9.84                       | 8.61                                       |
| ALL   | 7,105,768   | **MAE (kg/min)** | 1.99        | 11.55                     | 4.58       | 3.44                       | 3.29                                       |
|       |            | **ME (kg/min)**  | 1.30       | -9.92                     | 0.84       | -2.03                      | -2.09                                      |
||||||||||
|       |            | **Processing time (s)** | 3          | 284                      | 255        | 474                        | 28                                         |

The Acropole model is a neural network trained on Quick Access Recorder (QAR) data from
several aircraft types. Evaluation and the list of supported aircraft are available at
[evaluation/Dense_Acropole_FuelFlow_Scaling](https://github.com/DGAC/Acropole/tree/main/evaluation/Dense_Acropole_FuelFlow_Scaling).

## Development

Set up a full development environment (runtime, dev, and docs groups):

```bash
uv sync --all-groups
```

Common tasks:

```bash
uv run pytest                    # run the test suite (98% coverage)
make lint                        # ruff + mypy strict on src AND tests
pre-commit install               # enable the pre-commit hooks
```

## License

Acropole is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
See the [LICENSE](LICENSE) file for the full text.

## Credits

```bibtex
@inproceedings{jarry2024towards,
  title={Towards aircraft generic Quick Access Recorder fuel flow regression models for ADS-B data},
  author={Jarry, Gabriel and Delahaye, Daniel and Hurter, Christophe},
  booktitle={International Conference on Research in Air Transportation},
  year={2024}
}
```
