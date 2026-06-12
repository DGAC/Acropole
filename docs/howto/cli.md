# Estimate fuel from the command line

You don't need to write any Python to estimate fuel flow. Installing `acropole` gives you
an `acropole` command that reads a flight file, estimates fuel flow, and **writes the
enriched table back to disk**.

For the full list of options, see the [CLI reference](../reference/cli.md).

## A simple CSV with standard column names

If your file already uses the standard names (`typecode`, `groundspeed`, `altitude`,
`vertical_rate`, …), just point the command at it:

```bash
acropole estimate flight.csv
```

This writes `flight_fuel.csv` next to the input — the original columns plus `fuel_flow`
(kg/s) and `fuel_flow_kgh` (kg/h). On success the command prints:

```text
wrote 1234 rows with fuel columns to flight_fuel.csv
```

## A CSV with your own column names

When your columns use different names, map each feature with the matching flag. Map only
what differs — anything you omit falls back to the standard name. The bundled QAR flight
uses non-standard names:

```bash
acropole estimate examples/example_flight.csv \
  --typecode FLPL_AIRC_TYPE --groundspeed GRND_SPD_KT --altitude ALTI_STD_FT \
  --vertical-rate VERT_SPD_FTMN --airspeed TRUE_AIR_SPD_KT --mass MASS_KG \
  --second FLIGHT_TIME --out result.csv
```

Passing `--second` (here the `FLIGHT_TIME` column) lets the model derive accelerations and
adds a `fuel_cumsum` column (cumulative kg burned).

## Write parquet instead of CSV

The output format follows the `--out` extension, independent of the input:

```bash
acropole estimate flight.csv --out flight_fuel.parquet
```

A parquet input works the same way — `acropole estimate flight.parquet` writes
`flight_fuel.parquet` by default.

## Process a batch of files

The command handles one file per call; loop in your shell to process many. Each file is
written next to itself with a `_fuel` suffix:

```bash
for f in flights/*.csv; do
  acropole estimate "$f"
done
```

Or write the results into a separate directory:

```bash
mkdir -p enriched
for f in flights/*.csv; do
  acropole estimate "$f" --out "enriched/$(basename "$f")"
done
```

!!! note "Errors stop the run"
    If a file is missing, has an unsupported format, or lacks a mapped column, the command
    prints the error to `stderr` and exits with status `1`. In a loop, add `|| continue`
    after the command to skip a bad file and keep going.
