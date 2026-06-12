# Command-Line Interface

Installing `acropole` installs the `acropole` command (entry point `acropole.cli:main`).
It is a thin wrapper over [`FuelEstimator`](index.md): read a flight from disk, estimate
fuel flow, and **write the enriched table back out**. The estimation logic lives entirely
in the library — the CLI only handles file I/O and argument parsing.

## `acropole estimate`

```text
acropole estimate <flight> [OPTIONS]
```

Read `<flight>` (`.csv` or `.parquet`), estimate fuel flow, and write the result. The
output frame is the input frame plus:

| Column | Unit | When |
|---|---|---|
| `fuel_flow` | kg/s | always |
| `fuel_flow_kgh` | kg/h | always |
| `fuel_cumsum` | kg | only when `--second` is given |

### Argument

| Argument | Description |
|---|---|
| `flight` | Path to the input flight, `.csv` or `.parquet`. |

### Options

| Option | Default | Description |
|---|---|---|
| `--out` | `<flight>_fuel.<ext>` | Output path (`.csv` or `.parquet`). When omitted, the result is written next to the input with a `_fuel` suffix, keeping the input's extension. |
| `--typecode` | `typecode` | Aircraft type column. |
| `--groundspeed` | `groundspeed` | Groundspeed column (kt). |
| `--altitude` | `altitude` | Altitude column (ft). |
| `--vertical-rate` | `vertical_rate` | Vertical rate column (ft/min). |
| `--airspeed` | `airspeed` | Airspeed column (kt). |
| `--mass` | `mass` | Mass column (kg). |
| `--second` | _(none)_ | Timestamp column (s). Enables temporal derivatives and the `fuel_cumsum` column. |

Each mapping option points a logical feature at the matching column name in your file. If
your file already uses the standard names, you can omit them all.

### Examples

Standard column names — the output defaults to `flight_fuel.csv`:

```bash
acropole estimate flight.csv
```

Choose an explicit output path:

```bash
acropole estimate flight.csv --out enriched.parquet
```

Real QAR flight with mapped columns (the bundled `examples/example_flight.csv`):

```bash
acropole estimate examples/example_flight.csv \
  --typecode FLPL_AIRC_TYPE --groundspeed GRND_SPD_KT --altitude ALTI_STD_FT \
  --vertical-rate VERT_SPD_FTMN --airspeed TRUE_AIR_SPD_KT --mass MASS_KG \
  --second FLIGHT_TIME --out result.csv
# wrote 7796 rows with fuel columns to result.csv
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success — the enriched table was written; the command prints `wrote <n> rows with fuel columns to <out>`. |
| `1` | Error — printed on `stderr`. Raised when the input file does not exist, the format is unsupported (not `.csv`/`.parquet`), or a mapped column is missing. |

For a task-oriented walkthrough, see the how-to guide
[Estimate fuel from the command line](../howto/cli.md).
