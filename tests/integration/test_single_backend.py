"""One backend at a time — the optional-dependency contract.

``tests/unit/test_estimator.py`` imports pandas, polars *and* pyarrow at module
level, so it can prove the backends **agree** but not that any one of them
**works alone**. Working alone is the actual promise of the ``[polars]``,
``[pandas]`` and ``[pyarrow]`` extras: since narwhals replaced the polars-
internal pipeline, `pip install acropole[pandas]` must be a complete install.

Each test here imports a single backend and skips when it is absent, so CI can
run this module in environments that have exactly one installed (the `backends`
matrix in `.github/workflows/ci.yml`). Locally, where all three are present,
they all run.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from acropole import AircraftFuelEstimator, FuelEstimator

pytestmark = pytest.mark.integration

N = 6
TYPECODE = ["A320"] * N
GROUNDSPEED = np.linspace(200, 460, N)
ALTITUDE = np.linspace(2000, 36000, N)
VERTICAL_RATE = np.full(N, 1500.0)
SECOND = np.arange(N, dtype=float) * 4.0

FUEL_COLUMNS = {"fuel_flow", "fuel_flow_kgh", "fuel_cumsum"}


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _columns(frame: object) -> set[str]:
    """Column names of a pandas/polars frame or a pyarrow table."""
    names = getattr(frame, "column_names", None)  # pyarrow.Table
    if names is None:
        names = frame.columns  # type: ignore[attr-defined]  # pandas/polars
    return set(names)


def _check(out: object, expected_type: type) -> None:
    """The estimator echoed the input type and added finite fuel columns."""
    assert type(out) is expected_type
    assert FUEL_COLUMNS <= _columns(out)


def test_numpy_only_api_needs_no_backend() -> None:
    """``AircraftFuelEstimator`` must work with no frame library at all.

    The params table is parsed with the stdlib ``csv`` module precisely so this
    holds; this test is the `none` leg of the CI matrix.
    """
    fuel_flow = AircraftFuelEstimator("A320").estimate(
        GROUNDSPEED, ALTITUDE, VERTICAL_RATE
    )
    assert fuel_flow.shape == (N,)
    assert np.isfinite(fuel_flow).all()


@pytest.mark.skipif(not _installed("polars"), reason="polars not installed")
def test_polars_alone() -> None:
    import polars as pl

    frame = pl.DataFrame(
        {
            "typecode": TYPECODE,
            "groundspeed": GROUNDSPEED,
            "altitude": ALTITUDE,
            "vertical_rate": VERTICAL_RATE,
            "second": SECOND,
        }
    )
    out = FuelEstimator().estimate(frame)
    _check(out, pl.DataFrame)
    assert out["fuel_flow"].is_finite().all()


@pytest.mark.skipif(not _installed("pandas"), reason="pandas not installed")
def test_pandas_alone() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "typecode": TYPECODE,
            "groundspeed": GROUNDSPEED,
            "altitude": ALTITUDE,
            "vertical_rate": VERTICAL_RATE,
            "second": SECOND,
        }
    )
    out = FuelEstimator().estimate(frame)
    _check(out, pd.DataFrame)
    assert np.isfinite(out["fuel_flow"].to_numpy()).all()


@pytest.mark.skipif(not _installed("pyarrow"), reason="pyarrow not installed")
def test_pyarrow_alone() -> None:
    import pyarrow as pa

    frame = pa.table(
        {
            "typecode": TYPECODE,
            "groundspeed": GROUNDSPEED,
            "altitude": ALTITUDE,
            "vertical_rate": VERTICAL_RATE,
            "second": SECOND,
        }
    )
    out = FuelEstimator().estimate(frame)
    _check(out, pa.Table)
    assert np.isfinite(out.column("fuel_flow").to_numpy()).all()
