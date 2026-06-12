"""Aircraft fuel-flow estimation from trajectory data via an ONNX model.

Public API:
    FuelEstimator         — DataFrame-in / DataFrame-out estimator (pandas or
                            polars accepted; the input type is preserved on
                            output). Dispatches per aircraft typecode internally.
    AircraftFuelEstimator — typecode-bound, numpy-only estimator (faster per call,
                            no per-call params lookup).
"""

from __future__ import annotations

import warnings
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import polars as pl

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["AircraftFuelEstimator", "FuelEstimator"]

# Intermediate numpy arrays carry the estimator's working precision (float32 or
# float64); float64 is the widest, so it types the public surface without Any.
type FloatArray = npt.NDArray[np.float64]
type FloatDType = np.dtype[np.float64]

# Normalization bounds, one entry per model input feature (order matters):
# engine_type, d_altitude, d_groundspeed, d_airspeed, surface, max_ope_alti,
# max_ope_speed, altitude, groundspeed, airspeed, vertical_rate, mass_norm.
_MAXIMUMS = [1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1]
_MINIMUMS = [0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0]

_DEFAULT_MASS = -1.0

# ONNX model emits a 2-D (N, 1) tensor; squeeze the trailing singleton axis.
_MODEL_OUTPUT_NDIM = 2


def diff_bfill(arr: FloatArray) -> FloatArray:
    """numpy equivalent of ``pandas.Series.diff().bfill()``.

    The first element repeats the first valid difference (index 1). For a
    single-element array there is no difference to back-fill, so the result is
    ``NaN`` — matching pandas exactly (``Series([x]).diff().bfill() == [NaN]``).
    """
    n = arr.shape[0]
    out = np.empty(n, dtype=arr.dtype)
    if n == 0:
        return out
    if n == 1:
        out[0] = np.nan
        return out
    out[1:] = arr[1:] - arr[:-1]
    out[0] = out[1]
    return out


def safe_divide(num: FloatArray, den: FloatArray) -> FloatArray:
    """Element-wise division that yields NaN (not inf) where the denominator is 0.

    Duplicate consecutive timestamps produce ``dt == 0``; rather than emit an
    ``inf`` that silently poisons every downstream cumulative sum, we surface a
    NaN at that sample so it is visibly missing.
    """
    out = np.full_like(num, np.nan)
    nonzero = den != 0
    np.divide(num, den, out=out, where=nonzero)
    return out


class AircraftFuelEstimator:
    """Typecode-bound fuel-flow estimator with numpy-only I/O.

    Faster than :meth:`FuelEstimator.estimate` per call because the typecode
    params lookup, normalization arrays and engine-scale factor are precomputed
    at construction. Input is 1-D numpy arrays; output is a 1-D numpy array of
    fuel flow in **kg/s** (multiply by 3600 for kg/h).
    """

    DEFAULT_MASS = _DEFAULT_MASS

    def __init__(
        self,
        typecode: str,
        aircraft_params_path: Annotated[
            str | None, "CSV path; None -> package data"
        ] = None,
        model_path: Annotated[str | None, "ONNX path; None -> package data"] = None,
        dtype: Annotated[npt.DTypeLike, "intermediate numpy precision"] = np.float64,
    ) -> None:
        if aircraft_params_path is None:
            aircraft_params_path = str(
                files("acropole").joinpath("data/aircraft_params.csv")
            )
        params = pl.read_csv(aircraft_params_path)
        row = params.filter(pl.col("ACFT_ICAO_TYPE") == typecode)
        if row.is_empty():
            raise ValueError(f"Aircraft type {typecode!r} not in aircraft_params")

        if model_path is None:
            model_path = str(
                files("acropole").joinpath("models/acropole_fuel_model.onnx")
            )
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        self._init_from(
            typecode, row.to_dicts()[0], session, cast("FloatDType", np.dtype(dtype))
        )

    @classmethod
    def _from_shared(
        cls, estimator: FuelEstimator, typecode: str
    ) -> AircraftFuelEstimator:
        """Build bound to ``typecode``, reusing ``estimator``'s session/params."""
        params = estimator._params_by_type.get(typecode)
        if params is None:
            raise ValueError(f"Aircraft type {typecode!r} not in aircraft_params")
        obj = cls.__new__(cls)
        obj._init_from(typecode, params, estimator.session, estimator.dtype)
        return obj

    def _init_from(  # type: ignore[no-any-unimported]  # ort.InferenceSession: no stubs
        self,
        typecode: str,
        params: dict[str, object],  # polars row: heterogeneous CSV values
        session: ort.InferenceSession,
        dtype: FloatDType,
    ) -> None:
        self.typecode = typecode
        self.dtype = dtype
        self.session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

        # params holds a polars CSV row (dict[str, object]); the queried columns
        # are numeric by construction, so the float coercions below are sound —
        # the ignores silence object-operand noise, not a real type risk.
        to_dtype = dtype.type
        self._engine_type = to_dtype(params["ENGINE_TYPE"])  # type: ignore[arg-type]
        self._surface = to_dtype(params["SURFACE"])  # type: ignore[arg-type]
        self._max_ope_alti = to_dtype(params["MAX_OPE_ALTI"])  # type: ignore[arg-type]
        self._max_ope_speed = to_dtype(params["MAX_OPE_SPEED"])  # type: ignore[arg-type]
        self._ope_empty_weight = to_dtype(params["OPE_EMPTY_WEIGHT"])  # type: ignore[arg-type]
        self._mass_range = to_dtype(
            params["MAX_TO_WEIGHT"] - params["OPE_EMPTY_WEIGHT"]  # type: ignore[operator]
        )
        self._fuel_scale = float(
            params["FUEL_FLOW_TO"] * params["ENGINE_NUM"]  # type: ignore[operator]
        )

        self._mins = np.array(_MINIMUMS, dtype=dtype)
        self._scale = np.array(_MAXIMUMS, dtype=dtype) - self._mins

    def estimate(
        self,
        groundspeed: FloatArray,
        altitude: FloatArray,
        vertical_rate: FloatArray,
        *,
        airspeed: FloatArray | None = None,
        mass: FloatArray | None = None,
        second: FloatArray | None = None,
        d_altitude: FloatArray | None = None,
        d_groundspeed: FloatArray | None = None,
        d_airspeed: FloatArray | None = None,
    ) -> FloatArray:
        """Predict per-sample fuel flow in **kg/s** (1-D array of length N)."""
        dtype = self.dtype
        gs = np.asarray(groundspeed, dtype=dtype)
        alt = np.asarray(altitude, dtype=dtype)
        vr = np.asarray(vertical_rate, dtype=dtype)
        n = gs.shape[0]
        air = np.asarray(airspeed, dtype=dtype) if airspeed is not None else gs

        mass_norm = self._mass_norm(mass, n, dtype)
        d_alt, d_gs, d_as = self._derivatives(
            n, dtype, alt, gs, air, vr, second, d_altitude, d_groundspeed, d_airspeed
        )

        inputs = np.empty((n, 12), dtype=dtype)
        inputs[:, 0] = self._engine_type
        inputs[:, 1] = d_alt
        inputs[:, 2] = d_gs
        inputs[:, 3] = d_as
        inputs[:, 4] = self._surface
        inputs[:, 5] = self._max_ope_alti
        inputs[:, 6] = self._max_ope_speed
        inputs[:, 7] = alt
        inputs[:, 8] = gs
        inputs[:, 9] = air
        inputs[:, 10] = vr
        inputs[:, 11] = mass_norm

        normalized = (inputs - self._mins) / self._scale
        if normalized.dtype != np.float32:
            normalized = normalized.astype(np.float32)

        values = self.session.run([self._output_name], {self._input_name: normalized})[
            0
        ]
        single = (
            values.squeeze(axis=-1)
            if values.ndim == _MODEL_OUTPUT_NDIM
            else values.ravel()
        )
        return np.asarray(single * self._fuel_scale)

    def _mass_norm(
        self, mass: FloatArray | None, n: int, dtype: FloatDType
    ) -> FloatArray:
        if mass is None:
            return np.full(n, self.DEFAULT_MASS, dtype=dtype)
        if self._mass_range == 0:
            warnings.warn(
                f"Aircraft {self.typecode!r} has MAX_TO_WEIGHT == OPE_EMPTY_WEIGHT; "
                "mass normalization is undefined (NaN).",
                stacklevel=2,
            )
            return np.full(n, np.nan, dtype=dtype)
        out = (
            np.asarray(mass, dtype=dtype) - self._ope_empty_weight
        ) / self._mass_range
        return np.asarray(out)

    def _derivatives(
        self,
        n: int,
        dtype: FloatDType,
        alt: FloatArray,
        gs: FloatArray,
        air: FloatArray,
        vr: FloatArray,
        second: FloatArray | None,
        d_altitude: FloatArray | None,
        d_groundspeed: FloatArray | None,
        d_airspeed: FloatArray | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        if second is None:
            d_alt = (
                np.asarray(d_altitude, dtype=dtype)
                if d_altitude is not None
                else vr / dtype.type(60.0)
            )
            d_gs = (
                np.asarray(d_groundspeed, dtype=dtype)
                if d_groundspeed is not None
                else np.zeros(n, dtype=dtype)
            )
            d_as = (
                np.asarray(d_airspeed, dtype=dtype)
                if d_airspeed is not None
                else np.zeros(n, dtype=dtype)
            )
            return d_alt, d_gs, d_as

        dt = diff_bfill(np.asarray(second, dtype=dtype))
        d_alt = (
            np.asarray(d_altitude, dtype=dtype)
            if d_altitude is not None
            else safe_divide(diff_bfill(alt), dt)
        )
        d_gs = (
            np.asarray(d_groundspeed, dtype=dtype)
            if d_groundspeed is not None
            else safe_divide(diff_bfill(gs), dt)
        )
        d_as = (
            np.asarray(d_airspeed, dtype=dtype)
            if d_airspeed is not None
            else safe_divide(diff_bfill(air), dt)
        )
        return d_alt, d_gs, d_as


class FuelEstimator:
    """Data pipeline for trajectory fuel-flow enhancement.

    Accepts a pandas **or** polars DataFrame and returns the same type, adding
    ``fuel_flow`` (kg/s), ``fuel_flow_kgh`` (kg/h) and — when ``second`` is given
    — ``fuel_cumsum`` (kg). Frames mixing several aircraft typecodes are handled
    per typecode (each row scored with its own aircraft parameters).
    """

    DEFAULT_MASS = _DEFAULT_MASS

    def __init__(
        self,
        aircraft_params_path: Annotated[
            str | None, "CSV path; None -> package data"
        ] = None,
        model_path: Annotated[str | None, "ONNX path; None -> package data"] = None,
        dtype: Annotated[npt.DTypeLike, "intermediate numpy precision"] = np.float64,
    ) -> None:
        if aircraft_params_path is None:
            aircraft_params_path = str(
                files("acropole").joinpath("data/aircraft_params.csv")
            )
        params = pl.read_csv(aircraft_params_path)
        self._params_by_type = {row["ACFT_ICAO_TYPE"]: row for row in params.to_dicts()}

        if model_path is None:
            model_path = str(
                files("acropole").joinpath("models/acropole_fuel_model.onnx")
            )
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.dtype: FloatDType = cast("FloatDType", np.dtype(dtype))

    def for_aircraft(self, typecode: str) -> AircraftFuelEstimator:
        """Return an :class:`AircraftFuelEstimator` bound to ``typecode``, reusing
        this estimator's already-loaded ONNX session and parameters (no reload)."""
        return AircraftFuelEstimator._from_shared(self, typecode)

    def estimate(
        self, flight: pd.DataFrame | pl.DataFrame, **kwargs: str
    ) -> pd.DataFrame | pl.DataFrame:
        """Estimate fuel flow for ``flight``; see class docstring for columns.

        Column-name overrides via kwargs: ``typecode``, ``groundspeed``,
        ``altitude``, ``vertical_rate``, ``airspeed``, ``mass``, ``second``,
        ``d_altitude``, ``d_groundspeed``, ``d_airspeed``.
        """
        df = flight if isinstance(flight, pl.DataFrame) else pl.from_pandas(flight)
        was_pandas = not isinstance(flight, pl.DataFrame)

        col: dict[str, str | None] = {
            name: kwargs.get(name, name)
            for name in (
                "typecode",
                "groundspeed",
                "altitude",
                "vertical_rate",
                "airspeed",
                "mass",
                "second",
                "d_altitude",
                "d_groundspeed",
                "d_airspeed",
            )
        }
        col["second"] = kwargs.get("second", None)

        for required in ("typecode", "groundspeed", "altitude", "vertical_rate"):
            if col[required] not in df.columns:
                raise ValueError(f"Column {col[required]!r} not found")
        if col["second"] is not None and df[col["second"]].dtype not in (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ):
            raise ValueError("column for second must be float or integer")

        fuel_flow = self._predict_grouped(df, col)

        out = df.with_columns(
            pl.Series("fuel_flow", fuel_flow),
            pl.Series("fuel_flow_kgh", fuel_flow * 3600.0),
        )
        second_col = col["second"]
        if second_col is not None:
            sec = df[second_col].to_numpy().astype(self.dtype)
            out = out.with_columns(
                pl.Series("fuel_cumsum", np.cumsum(fuel_flow * diff_bfill(sec)))
            )
        return out.to_pandas() if was_pandas else out

    def _predict_grouped(
        self, df: pl.DataFrame, col: dict[str, str | None]
    ) -> FloatArray:
        """Run inference per typecode group, scattering results back to row order."""
        typecode_col = col["typecode"] or "typecode"
        result = np.full(df.height, np.nan, dtype=self.dtype)
        typecodes = df[typecode_col]
        for typecode in typecodes.unique(maintain_order=True).to_list():
            mask = (typecodes == typecode).to_numpy()
            if typecode not in self._params_by_type:
                warnings.warn(f"Aircraft type {typecode!r} not supported", stacklevel=3)
                continue  # leave NaN for unsupported rows
            sub = df.filter(pl.lit(mask))
            groundspeed, altitude, vertical_rate, optional = self._extract(sub, col)
            result[mask] = self.for_aircraft(typecode).estimate(
                groundspeed, altitude, vertical_rate, **optional
            )
        return result

    def _extract(
        self, sub: pl.DataFrame, col: dict[str, str | None]
    ) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, FloatArray | None]]:
        """Split a typecode group into its required arrays and optional kwargs.

        The three required columns are guaranteed present by ``estimate``'s
        upfront validation, so they are returned non-optional; the rest become
        ``estimate``'s keyword-only arguments (``None`` when their column is
        absent).
        """
        dtype = self.dtype

        def arr(key: str) -> FloatArray | None:
            """Numpy array for column ``col[key]`` if present, else None."""
            name = col[key]
            if name is None or name not in sub.columns:
                return None
            return sub[name].to_numpy().astype(dtype)

        def required(key: str) -> FloatArray:
            """Array for a column ``estimate`` already validated as present."""
            name = col[key]
            if name is None:  # unreachable: validated in estimate() upfront
                raise ValueError(f"Column for {key!r} not found")
            return sub[name].to_numpy().astype(dtype)

        groundspeed = required("groundspeed")
        altitude = required("altitude")
        vertical_rate = required("vertical_rate")
        optional = {
            "airspeed": arr("airspeed"),
            "mass": arr("mass"),
            "second": arr("second"),
            "d_altitude": arr("d_altitude"),
            "d_groundspeed": arr("d_groundspeed"),
            "d_airspeed": arr("d_airspeed"),
        }
        return groundspeed, altitude, vertical_rate, optional
