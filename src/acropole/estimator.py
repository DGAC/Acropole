"""Aircraft fuel-flow estimation from trajectory data via an ONNX model.

Public API:
    FuelEstimator         — DataFrame-in / DataFrame-out estimator. Any frame
                            narwhals supports (pandas, polars, pyarrow, …) is
                            accepted and the input type is preserved on output.
                            Dispatches per aircraft typecode internally.
    AircraftFuelEstimator — typecode-bound, numpy-only estimator (faster per call,
                            no per-call params lookup).
"""

from __future__ import annotations

import csv
import warnings
from importlib.resources import files
from typing import Annotated, cast

# narwhals' *stable* namespace, not the top-level one: acropole is a library that
# gets installed next to arbitrary other packages, and stable.v1 is guaranteed not
# to break across narwhals major versions. See https://narwhals-dev.github.io/narwhals/backcompat/
import narwhals.stable.v1 as nw
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from narwhals.stable.v1.typing import IntoDataFrameT

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

# Columns of aircraft_params.csv the estimator reads, all numeric. The rest of
# the table (ENGINE_ICAO, TYPE, WTC, …) is reference metadata we never score on.
_PARAM_COLUMNS = (
    "ENGINE_TYPE",
    "SURFACE",
    "MAX_OPE_ALTI",
    "MAX_OPE_SPEED",
    "OPE_EMPTY_WEIGHT",
    "MAX_TO_WEIGHT",
    "FUEL_FLOW_TO",
    "ENGINE_NUM",
)


def _load_aircraft_params(path: str) -> dict[str, dict[str, float]]:
    """Read the aircraft reference table into ``{typecode: {param: value}}``.

    Parsed with the stdlib ``csv`` module rather than a dataframe backend: the
    table is a ~30-row lookup consumed as plain dicts, so reading it through
    polars/pandas would make a frame backend mandatory just to build a mapping.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            row["ACFT_ICAO_TYPE"]: {name: float(row[name]) for name in _PARAM_COLUMNS}
            for row in csv.DictReader(handle)
        }


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
        row = _load_aircraft_params(aircraft_params_path).get(typecode)
        if row is None:
            raise ValueError(f"Aircraft type {typecode!r} not in aircraft_params")

        if model_path is None:
            model_path = str(
                files("acropole").joinpath("models/acropole_fuel_model.onnx")
            )
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        self._init_from(typecode, row, session, cast("FloatDType", np.dtype(dtype)))

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
        params: dict[str, float],
        session: ort.InferenceSession,
        dtype: FloatDType,
    ) -> None:
        self.typecode = typecode
        self.dtype = dtype
        self.session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

        to_dtype = dtype.type
        self._engine_type = to_dtype(params["ENGINE_TYPE"])
        self._surface = to_dtype(params["SURFACE"])
        self._max_ope_alti = to_dtype(params["MAX_OPE_ALTI"])
        self._max_ope_speed = to_dtype(params["MAX_OPE_SPEED"])
        self._ope_empty_weight = to_dtype(params["OPE_EMPTY_WEIGHT"])
        self._mass_range = to_dtype(
            params["MAX_TO_WEIGHT"] - params["OPE_EMPTY_WEIGHT"]
        )
        self._fuel_scale = float(params["FUEL_FLOW_TO"] * params["ENGINE_NUM"])

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

    Accepts any eager DataFrame narwhals supports (pandas, polars, pyarrow, …)
    and returns the same type, adding
    ``fuel_flow`` (kg/s), ``fuel_flow_kgh`` (kg/h) and — when a ``second``
    column is present — ``fuel_cumsum`` (kg). The ``second`` column (like the
    other optional features) is triggered by its presence in the frame, no
    keyword argument required. Frames mixing several aircraft typecodes are
    handled per typecode (each row scored with its own aircraft parameters).
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
        self._params_by_type = _load_aircraft_params(aircraft_params_path)

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

    def estimate(self, flight: IntoDataFrameT, **kwargs: str) -> IntoDataFrameT:
        """Estimate fuel flow for ``flight``; see class docstring for columns.

        Optional features are triggered by the **presence of their column** in
        ``flight`` (matched by name): when a ``second`` column is present the
        real time-derivatives are used and a ``fuel_cumsum`` column is added; an
        absent ``second`` column falls back to quasi-steady derivatives and adds
        no ``fuel_cumsum`` (no kwarg required either way).

        Column-name overrides via kwargs map a non-standard column name onto a
        feature: ``typecode``, ``groundspeed``, ``altitude``, ``vertical_rate``,
        ``airspeed``, ``mass``, ``second``, ``d_altitude``, ``d_groundspeed``,
        ``d_airspeed``.
        """
        df = nw.from_native(flight, eager_only=True)

        col: dict[str, str] = {
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

        for required in ("typecode", "groundspeed", "altitude", "vertical_rate"):
            if col[required] not in df.columns:
                raise ValueError(f"Column {col[required]!r} not found")
        second_col = col["second"]
        if second_col in df.columns and not df[second_col].dtype.is_numeric():
            raise ValueError("column for second must be float or integer")

        fuel_flow = self._predict_grouped(df, col)

        backend = df.implementation
        out = df.with_columns(
            nw.new_series("fuel_flow", fuel_flow, backend=backend),
            nw.new_series("fuel_flow_kgh", fuel_flow * 3600.0, backend=backend),
        )
        if second_col in df.columns:
            sec = df[second_col].to_numpy().astype(self.dtype)
            out = out.with_columns(
                nw.new_series(
                    "fuel_cumsum",
                    np.cumsum(fuel_flow * diff_bfill(sec)),
                    backend=backend,
                )
            )
        return out.to_native()

    def _predict_grouped(
        self, df: nw.DataFrame[IntoDataFrameT], col: dict[str, str]
    ) -> FloatArray:
        """Run inference per typecode group, scattering results back to row order.

        Feature columns are pulled out of the frame once, up front; the per-group
        split is then a numpy mask. That keeps the backend-specific work to a
        single conversion per column instead of one frame filter per typecode.
        """
        typecode_col = col["typecode"]
        typecodes = df[typecode_col].to_numpy()
        required, optional = self._extract(df, col)

        result = np.full(len(df), np.nan, dtype=self.dtype)
        for typecode in dict.fromkeys(typecodes.tolist()):  # unique, order-preserving
            if typecode not in self._params_by_type:
                warnings.warn(f"Aircraft type {typecode!r} not supported", stacklevel=3)
                continue  # leave NaN for unsupported rows
            mask = typecodes == typecode
            groundspeed, altitude, vertical_rate = (arr[mask] for arr in required)
            result[mask] = self.for_aircraft(typecode).estimate(
                groundspeed,
                altitude,
                vertical_rate,
                **{k: None if v is None else v[mask] for k, v in optional.items()},
            )
        return result

    def _extract(
        self, df: nw.DataFrame[IntoDataFrameT], col: dict[str, str]
    ) -> tuple[tuple[FloatArray, FloatArray, FloatArray], dict[str, FloatArray | None]]:
        """Split the frame into its required feature arrays and optional kwargs.

        The three required columns are guaranteed present by ``estimate``'s
        upfront validation, so they are returned non-optional; the rest become
        ``estimate``'s keyword-only arguments (``None`` when their column is
        absent).
        """
        dtype = self.dtype

        def arr(key: str) -> FloatArray | None:
            """Numpy array for column ``col[key]`` if present, else None."""
            name = col[key]
            if name not in df.columns:
                return None
            return df[name].to_numpy().astype(dtype)

        def required(key: str) -> FloatArray:
            """Array for a column ``estimate`` already validated as present."""
            return df[col[key]].to_numpy().astype(dtype)

        features = (
            required("groundspeed"),
            required("altitude"),
            required("vertical_rate"),
        )
        optional = {
            "airspeed": arr("airspeed"),
            "mass": arr("mass"),
            "second": arr("second"),
            "d_altitude": arr("d_altitude"),
            "d_groundspeed": arr("d_groundspeed"),
            "d_airspeed": arr("d_airspeed"),
        }
        return features, optional
