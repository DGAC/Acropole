# Analyse a complete flight

This guide walks through a full flight — taxi, climb, cruise, descent — loaded
from the `example_flight.csv` shipped under `examples/`, visualises its
trajectory, estimates the fuel flow, and compares the estimate against the
**measured** fuel flow recorded in the file. It then shows how the estimate
degrades gracefully as optional features are removed.

It is the documentation counterpart of the `examples/fuel_estimation.ipynb`
notebook.

!!! note "Dependencies"
    The plots below use `matplotlib` and `pandas`, which are not core
    dependencies of `acropole`. Install them alongside the `pandas` extra:

    ```bash
    uv add "acropole[pandas]" matplotlib
    ```

## 1. Load and resample the flight

The model expects a sampling rate of about **4 seconds**. The example flight is
recorded at 1 Hz, so we resample with `iloc[::4]`:

```python
import pandas as pd
from matplotlib import pyplot as plt

pd.options.display.max_columns = 999

flight = pd.read_csv("example_flight.csv")
flight = flight.iloc[::4].reset_index(drop=True)
flight.head()
```

The columns follow a QAR naming scheme (`FLPL_AIRC_TYPE`, `GRND_SPD_KT`,
`ALTI_STD_FT`, …); we map them to the model inputs with keyword arguments in
the next step. The file also carries `FUEL_FLOW_KGH`, the **measured** per-engine
fuel flow, which we use as ground truth.

## 2. Visualise speed and altitude

```python
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

ax1.plot(flight.FLIGHT_TIME, flight.GRND_SPD_KT, color="b", lw=1, label="Ground Speed")
ax1.plot(flight.FLIGHT_TIME, flight.TRUE_AIR_SPD_KT, color="g", lw=1, label="True Air Speed")
ax1.legend()
ax1.set_xlabel("Flight time (s)")
ax1.set_ylabel("Speed (kt)")

ax2.plot(flight.FLIGHT_TIME, flight.ALTI_STD_FT, color="b", lw=1)
ax2.set_xlabel("Flight time (s)")
ax2.set_ylabel("Altitude standard (ft)")

ax3.plot(flight.FLIGHT_TIME, flight.VERT_SPD_FTMN, color="b", lw=1)
ax3.set_xlabel("Flight time (s)")
ax3.set_ylabel("Vertical Speed (ft/min)")

plt.tight_layout()
```

## 3. Estimate fuel flow (all features)

With timestamp, true airspeed and mass available, the estimate is at its most
accurate:

```python
from acropole import FuelEstimator

fe = FuelEstimator()

flight_fuel = fe.estimate(
    flight,
    typecode="FLPL_AIRC_TYPE",
    groundspeed="GRND_SPD_KT",
    altitude="ALTI_STD_FT",
    vertical_rate="VERT_SPD_FTMN",
    # optional features:
    second="FLIGHT_TIME",
    airspeed="TRUE_AIR_SPD_KT",
    mass="MASS_KG",
)
```

`estimate` adds `fuel_flow` (kg/s), `fuel_flow_kgh` (kg/h) and — because a
`second` column is present (here mapped from `FLIGHT_TIME`) — `fuel_cumsum`
(cumulative kg).

### Compare to the measured fuel flow

`acropole` predicts the **total** aircraft fuel flow (single-engine output
scaled by `ENGINE_NUM`), whereas `FUEL_FLOW_KGH` in the file is recorded
**per engine**. To compare like with like, scale the measured trace by the
engine count — `2` for this A320, but read it from your own data rather than
hard-coding it (a quad would be `4`):

```python
n_engines = 2  # A320 — use ENGINE_NUM for your aircraft

plt.plot(flight_fuel.FLIGHT_TIME, flight_fuel.FUEL_FLOW_KGH * n_engines, lw=1, label="actual (kg/h)")
plt.plot(flight_fuel.FLIGHT_TIME, flight_fuel.fuel_flow_kgh, alpha=0.8, lw=1, label="estimate (kg/h)")
plt.legend()
plt.show()
```

The estimate tracks the measured curve closely across all flight phases.

## 4. How optional features affect the estimate

The only **required** inputs are `typecode`, `groundspeed`, `altitude` and
`vertical_rate`. Everything else is optional and improves accuracy. Dropping
features one by one shows their contribution.

### Without mass

```python
flight_fuel = fe.estimate(
    flight,
    typecode="FLPL_AIRC_TYPE",
    groundspeed="GRND_SPD_KT",
    altitude="ALTI_STD_FT",
    vertical_rate="VERT_SPD_FTMN",
    second="FLIGHT_TIME",
    airspeed="TRUE_AIR_SPD_KT",
    # no mass
)
```

### Without mass nor airspeed

When `airspeed` is omitted it defaults to `groundspeed`:

```python
flight_fuel = fe.estimate(
    flight,
    typecode="FLPL_AIRC_TYPE",
    groundspeed="GRND_SPD_KT",
    altitude="ALTI_STD_FT",
    vertical_rate="VERT_SPD_FTMN",
    second="FLIGHT_TIME",
    # no airspeed, no mass
)
```

### Minimum information

With only the four required columns, the derivatives fall back to a
quasi-steady-state assumption (`d_groundspeed = d_airspeed = 0`,
`d_altitude = vertical_rate / 60`):

```python
flight_fuel = fe.estimate(
    flight,
    typecode="FLPL_AIRC_TYPE",
    groundspeed="GRND_SPD_KT",
    altitude="ALTI_STD_FT",
    vertical_rate="VERT_SPD_FTMN",
)
```

Each removed feature widens the gap with the measured curve, but the estimate
remains usable even with the minimum input set. Provide `second` whenever you
can — it unlocks the speed/altitude derivatives that drive accuracy during
acceleration and climb.

## See also

- [Map your own columns](columns-mapping.md)
- [Derivatives & the `second` column](derivatives.md)
- [Tutorial — Getting started](../tutorials/getting-started.md)
