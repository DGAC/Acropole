# Getting Started

This tutorial takes you from an empty environment to interpreting a real fuel-flow
estimation. By the end you will have run `FuelEstimator` on a small synthetic frame and
on the bundled example flight, and you will know what each output column means.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

The core library runs on numpy + polars + onnxruntime. The examples below use pandas,
so install the optional `pandas` extra:

```bash
pip install "acropole[pandas]"
```

Or, working from a clone with uv:

```bash
git clone https://github.com/DGAC/Acropole.git
cd Acropole
uv sync
```

Verify the install:

```python
from acropole import FuelEstimator

print(FuelEstimator())  # loads the bundled ONNX model and aircraft parameters
```

## Step 1 — Your first estimate

`FuelEstimator.estimate()` takes a `DataFrame` and returns the same frame with fuel-flow
columns appended. The four **required** columns are `typecode`, `groundspeed` (kt),
`altitude` (ft) and `vertical_rate` (ft/min):

```python
import pandas as pd
from acropole import FuelEstimator

fe = FuelEstimator()

flight = pd.DataFrame({
    "typecode": ["A320", "A320", "A320", "A320"],
    "groundspeed": [400, 410, 420, 430],     # kt
    "altitude": [10000, 11000, 12000, 13000],  # ft
    "vertical_rate": [2000, 1500, 1000, 500],   # ft/min
})

result = fe.estimate(flight)
print(result[["fuel_flow", "fuel_flow_kgh"]])
```

Because the input was a pandas `DataFrame`, the output is a pandas `DataFrame` too. Pass
a polars frame and you get a polars frame back.

## Step 2 — Add optional features

Supplying `airspeed`, `mass` and especially `second` improves the estimate. Each optional
feature is detected by the **presence of its column** in the frame (matched by name) — no
keyword argument is needed. The `second` column (elapsed seconds) lets the model derive
**accelerations** of the speeds and also produces a `fuel_cumsum` column (cumulative kg
burned):

```python
flight = pd.DataFrame({
    "typecode": ["A320", "A320", "A320", "A320"],
    "groundspeed": [400, 410, 420, 430],
    "altitude": [10000, 11000, 12000, 13000],
    "vertical_rate": [2000, 1500, 1000, 500],
    "second": [0.0, 4.0, 8.0, 12.0],          # present -> derivatives + fuel_cumsum
    "airspeed": [400, 410, 420, 430],
    "mass": [60000, 60000, 60000, 60000],
})

result = fe.estimate(flight)  # no kwarg needed: the `second` column triggers it
print(result[["fuel_flow", "fuel_flow_kgh", "fuel_cumsum"]])
```

!!! tip "Sampling rate"
    The model was trained on a **4-second** sampling rate. A much higher or lower rate
    yields noisier fuel flow — resample your trajectory to ~4 s first.

## Step 3 — Run on the example flight

The repository ships a real QAR flight at `examples/example_flight.csv`. Its column names
differ from the defaults, so map them with keyword arguments:

```python
import pandas as pd
from acropole import FuelEstimator

flight = pd.read_csv("examples/example_flight.csv")
flight = flight.iloc[::4]  # resample to ~4 s

fe = FuelEstimator()
result = fe.estimate(
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

## Step 4 — Interpret the output

`estimate()` appends three columns:

| Column | Unit | Meaning |
|---|---|---|
| `fuel_flow` | kg/s | Instantaneous fuel flow at each sample |
| `fuel_flow_kgh` | kg/h | The same value scaled by 3600 (kg/s → kg/h) |
| `fuel_cumsum` | kg | Cumulative fuel burned since the start (only when a `second` column is present) |

A quick sanity check — total fuel burned and mean flow:

```python
print(f"Mean fuel flow : {result['fuel_flow_kgh'].mean():.0f} kg/h")
print(f"Total burned   : {result['fuel_cumsum'].iloc[-1]:.1f} kg")
```

## Or from the command line

If you have your flight in a CSV or parquet file, you can skip Python entirely. Installing
`acropole` also installs the `acropole` command, which reads the file, estimates fuel flow,
and **writes the enriched table back to disk**:

```bash
acropole estimate flight.csv   # writes flight_fuel.csv
```

See [Estimate fuel from the command line](../howto/cli.md) for column mapping, parquet
output and batch processing.

## Next Steps

- [How-To: map your own columns](../howto/columns-mapping.md)
- [How-To: multi-aircraft frames](../howto/multi-aircraft.md)
- [How-To: derivatives vs `second`](../howto/derivatives.md)
- [Explanation: architecture](../explanation/architecture.md)
- [Reference: Python API](../reference/index.md)
