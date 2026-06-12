# Estimate multi-aircraft frames

A single `DataFrame` can mix several aircraft typecodes. `FuelEstimator.estimate()`
groups rows by typecode and scores each group with **its own aircraft parameters**, then
scatters the results back into the original row order — so you can process a whole fleet
in one call.

## A mixed frame

```python
import pandas as pd
from acropole import FuelEstimator

fe = FuelEstimator()

flight = pd.DataFrame({
    "typecode": ["A320", "A320", "B738", "B738"],
    "groundspeed": [400, 410, 420, 430],
    "altitude": [10000, 11000, 12000, 13000],
    "vertical_rate": [2000, 1500, 1000, 500],
})

result = fe.estimate(flight)
# A320 rows scored with A320 params, B738 rows with B738 params
```

## Unsupported typecodes

If a typecode is not present in the bundled aircraft parameters, those rows are left as
`NaN` in `fuel_flow` and a warning is emitted — the rest of the frame is still scored:

```python
import warnings

flight = pd.DataFrame({
    "typecode": ["A320", "ZZZZ"],
    "groundspeed": [400, 410],
    "altitude": [10000, 11000],
    "vertical_rate": [2000, 1500],
})

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = fe.estimate(flight)

print(result["fuel_flow"])      # second row is NaN
print([str(w.message) for w in caught])  # "Aircraft type 'ZZZZ' not supported"
```

## Sharing the session across aircraft

When you repeatedly score one aircraft at a time, use `for_aircraft()` to obtain a
typecode-bound `AircraftFuelEstimator` that **reuses the already-loaded ONNX session and
parameters** (no reload per typecode):

```python
fe = FuelEstimator()
a320 = fe.for_aircraft("A320")
b738 = fe.for_aircraft("B738")  # same session, different params
```

See the [Reference](../reference/index.md) for `AircraftFuelEstimator`'s numpy-only
`estimate()` signature.
