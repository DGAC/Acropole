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
from typing import Literal

import narwhals.stable.v1 as nw
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


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_missing_typecode_backend_parity(
    backend: Literal["pandas", "polars", "pyarrow"],
) -> None:
    """Without timestamps, missing typecodes only invalidate their own rows."""
    pytest.importorskip(backend)
    frame = nw.from_dict(
        {
            "typecode": ["A320", None, *TYPECODE[2:]],
            "groundspeed": GROUNDSPEED,
            "altitude": ALTITUDE,
            "vertical_rate": VERTICAL_RATE,
        },
        backend=backend,
    ).to_native()
    expected = AircraftFuelEstimator("A320").estimate(
        GROUNDSPEED, ALTITUDE, VERTICAL_RATE
    )
    with pytest.warns(UserWarning, match="not supported"):
        out = nw.from_native(FuelEstimator().estimate(frame), eager_only=True)
    actual = out["fuel_flow"].to_numpy()
    known = np.arange(N) != 1
    assert actual.shape == (N,)
    assert np.isnan(actual[1])
    np.testing.assert_allclose(actual[known], expected[known])


@pytest.mark.parametrize("dtype", ["string", "string[pyarrow]"])
def test_pandas_nullable_typecode(dtype: str) -> None:
    """Nullable pandas strings must not make the numpy group mask ambiguous."""
    pytest.importorskip("pandas")
    if dtype == "string[pyarrow]":
        pytest.importorskip("pyarrow")
    import pandas as pd

    frame = pd.DataFrame(
        {
            "typecode": pd.Series(["A320", None, *TYPECODE[2:]], dtype=dtype),
            "groundspeed": GROUNDSPEED,
            "altitude": ALTITUDE,
            "vertical_rate": VERTICAL_RATE,
        }
    )
    expected = AircraftFuelEstimator("A320").estimate(
        GROUNDSPEED, ALTITUDE, VERTICAL_RATE
    )
    with pytest.warns(UserWarning, match="not supported"):
        actual = FuelEstimator().estimate(frame)["fuel_flow"].to_numpy()
    known = np.arange(N) != 1
    assert np.isnan(actual[1])
    np.testing.assert_allclose(actual[known], expected[known])


@pytest.mark.parametrize("index_kind", ["offset", "duplicate", "datetime"])
def test_pandas_index_and_input_preserved(index_kind: str) -> None:
    """Predictions are positional while the caller retains its index and input."""
    pytest.importorskip("pandas")
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
    estimator = FuelEstimator()
    expected = estimator.estimate(frame)["fuel_flow"].to_numpy()
    if index_kind == "offset":
        frame.index = pd.Index(np.arange(N) + 10, name="sample")
    elif index_kind == "duplicate":
        frame.index = pd.Index(np.repeat(np.arange(N // 2), 2), name="sample")
    else:
        frame.index = pd.date_range("2026-01-01", periods=N, name="sample")
    original = frame.copy(deep=True)
    out = estimator.estimate(frame)
    pd.testing.assert_index_equal(out.index, frame.index)
    pd.testing.assert_frame_equal(frame, original)
    np.testing.assert_array_equal(out["fuel_flow"].to_numpy(), expected)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("gap", ["missing", "single-sample"])
def test_unknown_consumption_keeps_later_totals_unknown(
    backend: Literal["pandas", "polars", "pyarrow"], gap: str
) -> None:
    """A missing contribution cannot become a known cumulative fuel total."""
    pytest.importorskip(backend)
    codes = ["A320", None if gap == "missing" else "B738", *TYPECODE[2:]]
    frame = nw.from_dict(
        {
            "typecode": codes,
            "groundspeed": GROUNDSPEED,
            "altitude": ALTITUDE,
            "vertical_rate": VERTICAL_RATE,
            "second": SECOND,
        },
        backend=backend,
    ).to_native()
    estimator = FuelEstimator()
    if gap == "missing":
        with pytest.warns(UserWarning, match="not supported"):
            native_out = estimator.estimate(frame)
    else:
        native_out = estimator.estimate(frame)
    out = nw.from_native(native_out, eager_only=True)
    flow = out["fuel_flow"].to_numpy()
    cumulative = out["fuel_cumsum"].to_numpy()
    np.testing.assert_array_equal(np.isfinite(flow), np.arange(N) != 1)
    assert np.isfinite(cumulative[0])
    assert np.isnan(cumulative[1:]).all()
