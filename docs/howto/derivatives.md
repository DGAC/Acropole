# Provide derivatives or `second`

The model consumes three derivative features — the rate of change of altitude,
groundspeed and airspeed. Acropole gives you three ways to supply them, in increasing
order of control.

## Option A — let Acropole derive them from `second`

This is the recommended path. Provide a `second` column (elapsed seconds per sample) and
Acropole computes each derivative as a backward difference divided by the time step
(`Δfeature / Δt`):

```python
flight = pd.DataFrame({
    "typecode": ["A320", "A320", "A320"],
    "groundspeed": [400, 410, 420],
    "altitude": [10000, 11000, 12000],
    "vertical_rate": [2000, 1500, 1000],
    "second": [0.0, 4.0, 8.0],   # drives the derivatives
})

result = fe.estimate(flight)   # also produces fuel_cumsum
```

This path is the only one that also emits the `fuel_cumsum` column.

!!! note "Duplicate timestamps"
    Where two consecutive samples share a timestamp (`Δt == 0`), the derivative is set to
    `NaN` rather than `inf`, so a bad sample is visibly missing instead of silently
    poisoning the cumulative sum.

## Option B — no `second`, no derivatives

If you provide neither `second` nor pre-computed derivatives, Acropole falls back to a
reasonable default: `d_altitude` is derived from `vertical_rate` (converted ft/min → ft/s)
and the speed derivatives are treated as zero.

```python
flight = pd.DataFrame({
    "typecode": ["A320", "A320", "A320"],
    "groundspeed": [400, 410, 420],
    "altitude": [10000, 11000, 12000],
    "vertical_rate": [2000, 1500, 1000],
})

result = fe.estimate(flight)   # no fuel_cumsum, speed accelerations assumed 0
```

This is the least accurate path — prefer Option A whenever you have timestamps.

## Option C — supply pre-computed derivatives

If you already have accelerations from your own pipeline, pass them directly. Any
derivative you supply overrides what Acropole would compute; any you omit is derived from
`second` (if given) or falls back as in Option B:

```python
result = fe.estimate(
    flight,
    second="FLIGHT_TIME",
    d_altitude="DALT",
    d_groundspeed="DGS",
    d_airspeed="DAS",
)
```

This mixes freely: e.g. supply `d_groundspeed` only, and `d_altitude` / `d_airspeed` are
still derived from `second`.

## Summary

| Inputs | Altitude derivative | Speed derivatives | `fuel_cumsum` |
|---|---|---|---|
| `second` given | `Δalt / Δt` | `Δspeed / Δt` | ✅ |
| nothing given | `vertical_rate / 60` | `0` | ❌ |
| `d_*` given | uses your value | uses your value | ✅ if `second` given |
