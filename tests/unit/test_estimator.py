"""Unit tests for acropole.estimator (no real model I/O beyond the packaged ONNX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from acropole import FuelEstimator as FuelEstimatorPublic
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

MIN_CORRELATION = 0.97


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
        out = FuelEstimator().estimate(_flight(second=True), second="second")
        assert "fuel_cumsum" in out.columns
        assert out["fuel_cumsum"].is_monotonic_increasing

    def test_missing_required_column_raises(self) -> None:
        bad = _flight().drop(columns=["altitude"])
        with pytest.raises(ValueError, match="altitude"):
            FuelEstimator().estimate(bad)

    def test_unsupported_typecode_warns_and_nans(self) -> None:
        flight = _flight(typecode="ZZZZ")
        with pytest.warns(UserWarning, match="not supported"):
            out = FuelEstimator().estimate(flight)
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
        out = FuelEstimator().estimate(pl.from_pandas(_flight(second=True)), second="second")
        assert "fuel_cumsum" in out.columns

    def test_dup_timestamps_yield_nan_not_inf(self) -> None:
        flight = _flight(n=6, second=True)
        sec = flight["second"].to_numpy().copy()
        sec[3] = sec[2]  # dt == 0 at that point
        flight["second"] = sec
        out = FuelEstimator().estimate(flight, second="second")
        # safe_divide turns the dt==0 derivative into NaN, never inf
        assert not np.isinf(out["fuel_flow"].to_numpy()).any()


@pytest.mark.skipif(not EXAMPLE.exists(), reason="example_flight.csv not packaged")
class TestExampleFlight:
    """Real example-flight predictions vs measured fuel flow.

    Uses the packaged ONNX model and the example flight shipped under examples/.
    The CSV carries FUEL_FLOW_KGH (measured) — asserting the prediction tracks it
    guards against model-path / normalization regressions far better than a toy
    frame. Classified unit: the data is a local fixture, the model is packaged, no
    external boundary (network/DB/subprocess) is crossed.
    """

    @pytest.fixture(scope="class")
    def prediction(self) -> tuple[pd.Series, pd.Series]:
        flight = pd.read_csv(EXAMPLE).iloc[::4].reset_index(drop=True)
        out = FuelEstimatorPublic().estimate(flight, **MAPPING)
        return out["fuel_flow_kgh"], flight["FUEL_FLOW_KGH"]

    def test_no_nan(self, prediction: tuple[pd.Series, pd.Series]) -> None:
        pred, _ = prediction
        assert not pred.isna().any()

    def test_tracks_measured_fuel_flow(self, prediction: tuple[pd.Series, pd.Series]) -> None:
        pred, real = prediction
        # the predicted curve must match the measured fuel flow closely
        assert float(pred.corr(real)) > MIN_CORRELATION

    def test_cumsum_monotonic(self) -> None:
        flight = pd.read_csv(EXAMPLE).iloc[::4].reset_index(drop=True)
        out = FuelEstimatorPublic().estimate(flight, **MAPPING)
        assert out["fuel_cumsum"].is_monotonic_increasing
