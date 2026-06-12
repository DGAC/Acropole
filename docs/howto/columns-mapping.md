# Map your own column names

`FuelEstimator.estimate()` expects logical feature names (`typecode`, `groundspeed`,
`altitude`, `vertical_rate`, …). When your frame uses different column names, map them
with keyword arguments instead of renaming the frame.

## The default names

If your columns already use the default names, no mapping is needed:

```python
fe.estimate(flight)  # expects typecode, groundspeed, altitude, vertical_rate
```

## Overriding names

Each keyword argument maps a logical feature to the column name in your frame. Map only
what differs — anything you omit falls back to the default name:

```python
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

## Required vs optional features

| Keyword | Required | Unit | Notes |
|---|---|---|---|
| `typecode` | ✅ | — | ICAO aircraft type (e.g. `A320`) |
| `groundspeed` | ✅ | kt | |
| `altitude` | ✅ | ft | |
| `vertical_rate` | ✅ | ft/min | |
| `airspeed` | optional | kt | defaults to `groundspeed` if absent |
| `mass` | optional | kg | improves the estimate |
| `second` | optional | s | enables derivatives + `fuel_cumsum` |
| `d_altitude` | optional | — | pre-computed derivative (see [Derivatives](derivatives.md)) |
| `d_groundspeed` | optional | — | pre-computed derivative |
| `d_airspeed` | optional | — | pre-computed derivative |

## What gets validated

- The four required columns must exist under the mapped names, otherwise `estimate()`
  raises `ValueError: Column '<name>' not found`.
- If you map `second`, that column must be a numeric (float or integer) dtype, otherwise
  `estimate()` raises `ValueError: column for second must be float or integer`.

!!! note
    Unmapped optional features are simply skipped — passing a frame with no `mass`
    column is fine, the model uses its default mass handling.
