"""Unit tests for acropole.estimator (no real model I/O beyond the packaged ONNX)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import pytest

from acropole.estimator import (
    AircraftFuelEstimator,
    FuelEstimator,
    diff_bfill,
    safe_divide,
)

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "example_flight.csv"

MAPPING = {
    "typecode": "FLPL_AIRC_TYPE",
    "groundspeed": "GRND_SPD_KT",
    "altitude": "ALTI_STD_FT",
    "vertical_rate": "VERT_SPD_FTMN",
    "second": "FLIGHT_TIME",
    "airspeed": "TRUE_AIR_SPD_KT",
    "mass": "MASS_KG",
}

# example_flight.csv is an A320 (twin-engine); its FUEL_FLOW_KGH is recorded per
# engine, so the total is the measurement times ENGINE_NUM.
ENGINE_NUM = 2

# Golden reference: predicted fuel_flow_kgh at fixed row indices, captured from
# the validated ONNX model (numerically identical to the TensorFlow baseline,
# tag v0.1.0-alpha). A drift in the model, feature order or normalization breaks
# this. Regenerate intentionally only when the model itself changes.
GOLDEN_FUEL_FLOW_KGH = {
    100: 681.626,
    300: 5175.145,
    500: 3577.721,
    800: 2725.447,
    1200: 2672.506,
    1500: 2647.173,
    1900: 683.029,
}

# Measured MAPE on this flight (in-flight points) is ~5.3%; bound it generously
# but tightly enough to catch a real regression.
MAX_MAPE_PCT = 7.0

# Below this measured fuel flow (kg/h, total) the aircraft is idling on the
# ground; those points are excluded from the MAPE so taxi noise doesn't dominate.
IN_FLIGHT_THRESHOLD_KGH = 50.0


def _flight(n: int = 8, typecode: str = "A320", second: bool = False) -> pd.DataFrame:
    cols = {
        "typecode": [typecode] * n,
        "groundspeed": np.linspace(200, 460, n),
        "altitude": np.linspace(2000, 36000, n),
        "vertical_rate": np.full(n, 1500.0),
    }
    if second:
        cols["second"] = np.arange(n, dtype=float) * 4.0
    return pd.DataFrame(cols)


class TestDiffBfill:
    def test_matches_pandas_for_n_ge_2(self) -> None:
        arr = np.array([5.0, 9.0, 2.0, 11.0])
        expected = pd.Series(arr).diff().bfill().to_numpy()
        np.testing.assert_array_equal(diff_bfill(arr), expected)

    def test_single_element_is_nan_like_pandas(self) -> None:
        # pandas: Series([x]).diff().bfill() == [NaN]
        assert np.isnan(diff_bfill(np.array([42.0]))[0])

    def test_empty(self) -> None:
        assert diff_bfill(np.array([])).shape == (0,)


class TestSafeDivide:
    def test_zero_denominator_yields_nan_not_inf(self) -> None:
        out = safe_divide(np.array([1.0, 2.0]), np.array([0.0, 2.0]))
        assert np.isnan(out[0])
        assert out[1] == 1.0


class TestEstimateContract:
    def test_pandas_in_pandas_out(self) -> None:
        out = FuelEstimator().estimate(_flight())
        assert isinstance(out, pd.DataFrame)
        assert {"fuel_flow", "fuel_flow_kgh"} <= set(out.columns)

    def test_polars_in_polars_out(self) -> None:
        flight = _flight()
        out = FuelEstimator().estimate(pl.from_pandas(flight))
        assert isinstance(out, pl.DataFrame)
        # behavioral: same fuel columns, finite values, original rows preserved
        assert {"fuel_flow", "fuel_flow_kgh"} <= set(out.columns)
        assert out.height == len(flight)
        assert out["fuel_flow"].is_finite().all()

    def test_second_adds_cumsum(self) -> None:
        out = cast(
            "pd.DataFrame",
            FuelEstimator().estimate(_flight(second=True), second="second"),
        )
        assert "fuel_cumsum" in out.columns
        assert out["fuel_cumsum"].is_monotonic_increasing

    def test_missing_required_column_raises(self) -> None:
        bad = _flight().drop(columns=["altitude"])
        with pytest.raises(ValueError, match="altitude"):
            FuelEstimator().estimate(bad)

    def test_unsupported_typecode_warns_and_nans(self) -> None:
        flight = _flight(typecode="ZZZZ")
        with pytest.warns(UserWarning, match="not supported"):
            out = cast("pd.DataFrame", FuelEstimator().estimate(flight))
        assert out["fuel_flow"].isna().all()

    def test_multi_typecode_uses_per_aircraft_params(self) -> None:
        # Each typecode must be scored with its own params: stacking the same
        # rows under two typecodes must give two different fuel flows.
        a = _flight(typecode="A320")
        b = _flight(typecode="B738")
        mixed = pd.concat([a, b], ignore_index=True)
        out = FuelEstimator().estimate(mixed)
        ff_a = out["fuel_flow"].to_numpy()[: len(a)]
        ff_b = out["fuel_flow"].to_numpy()[len(a) :]
        assert not np.allclose(ff_a, ff_b)


class TestSecondPresenceTriggers:
    """A ``second`` column is detected by presence, no kwarg required (#15)."""

    def test_no_kwarg_present_second_adds_cumsum_pandas(self) -> None:
        # AC1: user case — second column present, no kwarg -> fuel_cumsum.
        out = cast("pd.DataFrame", FuelEstimator().estimate(_flight(second=True)))
        assert "fuel_cumsum" in out.columns
        assert out["fuel_cumsum"].is_monotonic_increasing

    def test_no_kwarg_present_second_adds_cumsum_polars(self) -> None:
        # AC1 (polars path): same presence-trigger on a polars frame.
        out = cast(
            "pl.DataFrame",
            FuelEstimator().estimate(pl.from_pandas(_flight(second=True))),
        )
        assert "fuel_cumsum" in out.columns
        cumsum = out["fuel_cumsum"].to_numpy()
        assert np.all(np.diff(cumsum) >= 0)

    def test_fuel_flow_parity_with_and_without_kwarg_pandas(self) -> None:
        # AC2: presence alone must drive the real derivatives -> identical
        # fuel_flow whether or not the redundant second="second" kwarg is passed.
        flight = _flight(second=True)
        implicit = cast("pd.DataFrame", FuelEstimator().estimate(flight))
        explicit = cast(
            "pd.DataFrame", FuelEstimator().estimate(flight, second="second")
        )
        np.testing.assert_array_equal(
            implicit["fuel_flow"].to_numpy(), explicit["fuel_flow"].to_numpy()
        )

    def test_fuel_flow_parity_with_and_without_kwarg_polars(self) -> None:
        # AC2 (polars path): same parity on a polars frame.
        flight = pl.from_pandas(_flight(second=True))
        implicit = cast("pl.DataFrame", FuelEstimator().estimate(flight))
        explicit = cast(
            "pl.DataFrame", FuelEstimator().estimate(flight, second="second")
        )
        np.testing.assert_array_equal(
            implicit["fuel_flow"].to_numpy(), explicit["fuel_flow"].to_numpy()
        )

    def test_present_second_uses_real_derivatives_not_quasi_steady(self) -> None:
        # AC2: a present second column must change the prediction vs the
        # no-second quasi-steady fallback (proves derivatives actually flow in).
        with_second = cast(
            "pd.DataFrame", FuelEstimator().estimate(_flight(second=True))
        )
        without_second = cast(
            "pd.DataFrame", FuelEstimator().estimate(_flight(second=False))
        )
        assert not np.allclose(
            with_second["fuel_flow"].to_numpy(),
            without_second["fuel_flow"].to_numpy(),
        )

    def test_no_second_column_works_without_cumsum(self) -> None:
        # AC3: a frame with no second column still works -> no fuel_cumsum,
        # finite fuel_flow, no error (unchanged behavior).
        out = cast("pd.DataFrame", FuelEstimator().estimate(_flight(second=False)))
        assert "fuel_cumsum" not in out.columns
        assert np.isfinite(out["fuel_flow"].to_numpy()).all()

    def test_explicit_custom_second_column_name(self) -> None:
        # AC4: an explicit kwarg still maps a non-standard column name.
        flight = _flight(second=True).rename(columns={"second": "elapsed_s"})
        out = cast("pd.DataFrame", FuelEstimator().estimate(flight, second="elapsed_s"))
        assert "fuel_cumsum" in out.columns
        assert out["fuel_cumsum"].is_monotonic_increasing

    def test_custom_named_second_column_without_kwarg_is_ignored(self) -> None:
        # AC4 (negative): a non-standard column name is NOT auto-detected; it is
        # only the default "second" name (or an explicit kwarg) that triggers.
        flight = _flight(second=True).rename(columns={"second": "elapsed_s"})
        out = cast("pd.DataFrame", FuelEstimator().estimate(flight))
        assert "fuel_cumsum" not in out.columns


class TestAircraftFuelEstimator:
    def test_unknown_typecode_raises(self) -> None:
        with pytest.raises(ValueError, match="not in aircraft_params"):
            AircraftFuelEstimator("ZZZZ")

    def test_numpy_io_shape(self) -> None:
        est = AircraftFuelEstimator("A320")
        ff = est.estimate(
            groundspeed=np.linspace(200, 460, 8),
            altitude=np.linspace(2000, 36000, 8),
            vertical_rate=np.full(8, 1500.0),
        )
        assert ff.shape == (8,)

    def test_for_aircraft_shares_session(self) -> None:
        fe = FuelEstimator()
        sub = fe.for_aircraft("A320")
        assert sub.session is fe.session

    def test_for_aircraft_unknown_typecode_raises(self) -> None:
        with pytest.raises(ValueError, match="not in aircraft_params"):
            FuelEstimator().for_aircraft("ZZZZ")

    def test_precomputed_derivatives_used(self) -> None:
        # Supplying d_* arrays must bypass the second-based derivation.
        est = AircraftFuelEstimator("A320")
        n = 8
        ff = est.estimate(
            groundspeed=np.linspace(200, 460, n),
            altitude=np.linspace(2000, 36000, n),
            vertical_rate=np.full(n, 1500.0),
            d_altitude=np.full(n, 25.0),
            d_groundspeed=np.full(n, 1.0),
            d_airspeed=np.full(n, 1.0),
        )
        assert ff.shape == (n,)
        assert np.isfinite(ff).all()

    def test_precomputed_derivatives_with_second(self) -> None:
        # d_* provided AND second given: d_* take precedence, no division.
        est = AircraftFuelEstimator("A320")
        n = 6
        ff = est.estimate(
            groundspeed=np.linspace(200, 460, n),
            altitude=np.linspace(2000, 36000, n),
            vertical_rate=np.full(n, 1500.0),
            second=np.arange(n, dtype=float) * 4.0,
            d_altitude=np.full(n, 25.0),
            d_groundspeed=np.full(n, 1.0),
            d_airspeed=np.full(n, 1.0),
        )
        assert np.isfinite(ff).all()


class TestEdgeCases:
    def test_second_non_numeric_dtype_raises(self) -> None:
        flight = _flight(second=True)
        flight["second"] = flight["second"].astype(str)
        with pytest.raises(ValueError, match="float or integer"):
            FuelEstimator().estimate(flight, second="second")

    def test_zero_mass_range_warns_and_nans(self) -> None:
        est = AircraftFuelEstimator("A320")
        est._mass_range = est.dtype.type(0)  # degenerate reference data
        with pytest.warns(UserWarning, match="undefined"):
            ff = est.estimate(
                groundspeed=np.linspace(200, 460, 5),
                altitude=np.linspace(2000, 36000, 5),
                vertical_rate=np.full(5, 1500.0),
                mass=np.full(5, 60000.0),
            )
        assert np.isnan(ff).all()

    def test_polars_cumsum(self) -> None:
        out = FuelEstimator().estimate(
            pl.from_pandas(_flight(second=True)), second="second"
        )
        assert "fuel_cumsum" in out.columns

    def test_dup_timestamps_yield_nan_not_inf(self) -> None:
        flight = _flight(n=6, second=True)
        sec = flight["second"].to_numpy().copy()
        sec[3] = sec[2]  # dt == 0 at that point
        flight["second"] = sec
        out = FuelEstimator().estimate(flight, second="second")
        # safe_divide turns the dt==0 derivative into NaN, never inf
        assert not np.isinf(out["fuel_flow"].to_numpy()).any()

    def test_single_row_flight(self) -> None:
        out = FuelEstimator().estimate(_flight(n=1))
        assert len(out) == 1
        assert np.isfinite(out["fuel_flow"].to_numpy()).all()

    def test_fuel_flow_kgh_is_3600_times_fuel_flow(self) -> None:
        out = FuelEstimator().estimate(_flight())
        np.testing.assert_allclose(
            out["fuel_flow_kgh"].to_numpy(),
            out["fuel_flow"].to_numpy() * 3600.0,
            rtol=1e-9,
        )

    def test_column_order_is_irrelevant(self) -> None:
        # mapping is by name, so a shuffled frame must give identical results
        flight = _flight()
        shuffled = flight[flight.columns[::-1]]
        a = FuelEstimator().estimate(flight)["fuel_flow"].to_numpy()
        b = FuelEstimator().estimate(shuffled)["fuel_flow"].to_numpy()
        np.testing.assert_array_equal(a, b)

    def test_float32_close_to_float64(self) -> None:
        flight = _flight(n=16)
        f64 = FuelEstimator(dtype=np.float64).estimate(flight)["fuel_flow"].to_numpy()
        f32 = FuelEstimator(dtype=np.float32).estimate(flight)["fuel_flow"].to_numpy()
        np.testing.assert_allclose(f32, f64, rtol=1e-3)

    def test_typecode_is_case_sensitive(self) -> None:
        # Same aircraft, two casings: uppercase matches the table, lowercase does
        # not. Contrasting the two is the point — typecodes are case-sensitive.
        fe = FuelEstimator()
        upper = fe.estimate(_flight(typecode="A320"))["fuel_flow"].to_numpy()
        assert np.isfinite(upper).all()  # "A320" is scored
        with pytest.warns(UserWarning):
            lower = fe.estimate(_flight(typecode="a320"))["fuel_flow"].to_numpy()
        assert np.isnan(lower).all()  # "a320" is not


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_flight.csv not packaged")
class TestExampleFlight:
    """End-to-end behaviour on the real example flight (fixture CSV).

    The packaged CSV carries the measured FUEL_FLOW_KGH; the golden assertion
    pins exact predicted values (the regression net for the TF→ONNX migration)
    and the MAPE bound checks the estimate stays close to the measurement —
    either catches a drift a mere correlation check would silently pass.
    """

    @pytest.fixture(scope="class")
    def out(self) -> pd.DataFrame:
        flight = pd.read_csv(EXAMPLE).iloc[::4].reset_index(drop=True)
        result = cast("pd.DataFrame", FuelEstimator().estimate(flight, **MAPPING))
        result["FUEL_FLOW_KGH"] = flight["FUEL_FLOW_KGH"]
        return result

    def test_no_nan(self, out: pd.DataFrame) -> None:
        assert not out["fuel_flow_kgh"].isna().any()

    def test_golden_values(self, out: pd.DataFrame) -> None:
        # Exact regression net: pinned predictions must not drift.
        pred = out["fuel_flow_kgh"].to_numpy()
        for idx, expected in GOLDEN_FUEL_FLOW_KGH.items():
            np.testing.assert_allclose(pred[idx], expected, rtol=1e-3)

    def test_mape_within_bound(self, out: pd.DataFrame) -> None:
        pred = out["fuel_flow_kgh"].to_numpy()
        real = out["FUEL_FLOW_KGH"].to_numpy() * ENGINE_NUM
        mask = real > IN_FLIGHT_THRESHOLD_KGH  # in-flight points only
        mape = float(np.mean(np.abs(pred[mask] - real[mask]) / real[mask]) * 100)
        assert mape < MAX_MAPE_PCT

    def test_cumsum_monotonic(self, out: pd.DataFrame) -> None:
        assert out["fuel_cumsum"].is_monotonic_increasing
