"""Command-line interface for acropole.

A thin wrapper over :class:`acropole.FuelEstimator`: read a flight from a CSV (or
parquet), estimate fuel flow, write the enriched table back out. The estimation
logic lives entirely in the library — this module only handles I/O and argument
parsing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
import narwhals.stable.v1 as nw
from narwhals.stable.v1.typing import IntoDataFrame

from acropole import FuelEstimator

app = cyclopts.App(
    name="acropole",
    help="Predict aircraft fuel flow from trajectory data.",
)

# The library itself is backend-agnostic, but reading a file needs a concrete
# one. Preference order, first installed wins; `acropole[cli]` pulls in polars.
type BackendName = Literal["polars", "pyarrow", "pandas"]
_BACKENDS: tuple[BackendName, ...] = ("polars", "pyarrow", "pandas")


def _backend() -> BackendName:
    """The first installed frame backend, for narwhals' file readers."""
    for name in _BACKENDS:
        if importlib.util.find_spec(name) is not None:
            return name
    print(
        "error: no dataframe backend installed "
        "(pip install 'acropole[cli]' for polars)",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _read(path: Path) -> nw.DataFrame[IntoDataFrame]:
    """Load a flight table from .csv or .parquet via any available backend."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return nw.read_parquet(path, backend=_backend())
    if suffix == ".csv":
        return nw.read_csv(path, backend=_backend())
    print(
        f"error: unsupported input format {suffix!r} (use .csv or .parquet)",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _write(frame: nw.DataFrame[IntoDataFrame], path: Path) -> None:
    if path.suffix.lower() == ".parquet":
        frame.write_parquet(path)
    else:
        frame.write_csv(path)


@app.command
def estimate(
    flight: Annotated[Path, cyclopts.Parameter(help="Input flight CSV or parquet")],
    *,
    out: Annotated[
        Path | None,
        cyclopts.Parameter(
            help="Output path (.csv/.parquet); default: <flight>_fuel.<ext>"
        ),
    ] = None,
    typecode: Annotated[
        str, cyclopts.Parameter(help="Aircraft type column")
    ] = "typecode",
    groundspeed: Annotated[
        str, cyclopts.Parameter(help="Groundspeed column (kt)")
    ] = "groundspeed",
    altitude: Annotated[
        str, cyclopts.Parameter(help="Altitude column (ft)")
    ] = "altitude",
    vertical_rate: Annotated[
        str, cyclopts.Parameter(help="Vertical rate column (ft/min)")
    ] = "vertical_rate",
    airspeed: Annotated[
        str, cyclopts.Parameter(help="Airspeed column (kt)")
    ] = "airspeed",
    mass: Annotated[str, cyclopts.Parameter(help="Mass column (kg)")] = "mass",
    second: Annotated[
        str | None, cyclopts.Parameter(help="Timestamp column (s); enables derivatives")
    ] = None,
) -> None:
    """Estimate fuel flow for a flight and write the enriched table.

    Adds ``fuel_flow`` (kg/s), ``fuel_flow_kgh`` (kg/h) and, when ``--second`` is
    given, ``fuel_cumsum`` (kg).
    """
    if not flight.exists():
        print(f"error: file not found: {flight}", file=sys.stderr)
        raise SystemExit(1)

    frame = _read(flight)
    mapping = {
        "typecode": typecode,
        "groundspeed": groundspeed,
        "altitude": altitude,
        "vertical_rate": vertical_rate,
        "airspeed": airspeed,
        "mass": mass,
    }
    if second is not None:
        mapping["second"] = second

    try:
        estimated = FuelEstimator().estimate(frame.to_native(), **mapping)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    # estimate() echoes its input frame type, whichever backend _read picked.
    result = nw.from_native(estimated, eager_only=True)

    if out is None:
        out = flight.with_name(f"{flight.stem}_fuel{flight.suffix}")
    _write(result, out)
    print(f"wrote {len(result)} rows with fuel columns to {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
