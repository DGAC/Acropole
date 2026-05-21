import warnings
from importlib.resources import files

import numpy as np
import onnxruntime as ort
import pandas as pd


def _diff_bfill(arr: np.ndarray) -> np.ndarray:
    """numpy equivalent of pandas Series.diff().bfill(): first element repeats the second."""
    n = arr.shape[0]
    out = np.empty(n, dtype=arr.dtype)
    if n == 0:
        return out
    out[1:] = arr[1:] - arr[:-1]
    out[0] = out[1] if n > 1 else arr.dtype.type(0)
    return out


class FuelEstimator:
    """
    Class that contains data pipelines for trajectory enhancement
    """

    DEFAULT_MASS = -1.0
    _MAXIMUMS = [1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1]
    _MINIMUMS = [0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0]

    def __init__(
        self,
        aircraft_params_path: str = None,
        model_path: str = None,
        dtype=np.float64,
    ):
        """
        Initializes the FuelEstimator.

        Args:
            aircraft_params_path (str): The path to the aircraft table. Default is None (use package data).
            model_path (str): The path to the prediction ONNX model. Default is None (use package data).
            dtype: Precision for intermediate numpy computations (default np.float64).
                The ONNX model still consumes float32; this only governs the upstream math.
                Passing np.float32 is ~50% faster on very large batches at the cost of
                ~0.02 kg/h drift on typical flights.
        """

        if aircraft_params_path is None:
            aircraft_params_path = files("acropole").joinpath(
                "data/aircraft_params.csv"
            )

        self.aircraft_params = pd.read_csv(aircraft_params_path)
        self._params_by_type = {
            tc: row
            for tc, row in self.aircraft_params.set_index("ACFT_ICAO_TYPE").iterrows()
        }

        if model_path is None:
            model_path = files("acropole").joinpath(
                "models/acropole_fuel_model.onnx"
            )

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        self.dtype = np.dtype(dtype)
        self.MAXIMUMS = np.array(self._MAXIMUMS, dtype=self.dtype)
        self.MINIMUMS = np.array(self._MINIMUMS, dtype=self.dtype)
        self._scale = self.MAXIMUMS - self.MINIMUMS

    def estimate(self, flight: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Estimates fuel flow and consumption based on the flight trajectory.

        The minimum set of features are:
            - flight (pd.DataFrame): The flight data as a pandas DataFrame.
            - typecode (str): The column name for the aircraft type code.
                Default is "typecode".
            - groundspeed (str): The column name for the groundspeed (in knot).
                Default is "groundspeed".
            - altitude (str): The column name for the altitude (in feet).
                Default is "altitude".
            - vertical_rate (str): The column name for the vertical rate (in feet/minutes).
                Default is "vertical_rate".

        Optional arguments are:
            - second (str): The column name for the timestamp (in seconds).
                Default is None. Enables exact derivatives via Δcolumn/Δsecond.
            - airspeed (str): The column name for the airspeed (in knot).
                Default is "airspeed".
            - mass (str): The column name for the mass (in kilogram).
                Default is "mass".
            - d_altitude (str): The column name for a precomputed altitude
                derivative (in feet/second). Default is "d_altitude". If the
                column is absent, the value is derived from `second` (if given)
                or from `vertical_rate / 60`.
            - d_groundspeed (str): The column name for a precomputed groundspeed
                derivative (in knot/second). Default is "d_groundspeed". If the
                column is absent, the value is derived from `second` or
                assumed to be zero (quasi-steady state).
            - d_airspeed (str): The column name for a precomputed airspeed
                derivative (in knot/second). Default is "d_airspeed". If the
                column is absent, the value is derived from `second` or
                assumed to be zero (quasi-steady state).

        Returns:
            pd.DataFrame: The flight data with additional estimated fuel parameters
            (`fuel_flow` in kg/s, `fuel_flow_kgh` in kg/h, and `fuel_cumsum` in kg
            when `second` is given).

        Note:
            - When `second` is provided, the fuel estimation is more accurate,
                especially due to **derivatives of speeds** (acceleration) used in the estimation.
            - `airspeed` is optional. If not provided, it is assumed to be equal
                to groundspeed. However, accurate airspeed is recommended for better estimation.
            - Expected sampling rate is 4 seconds; higher or lower sampling rate might induce
                noisier fuel flow. Resampling data before estimating fuel flow is recommended.

        Warnings:
            If the aircraft type code is not supported, returns NaN fuel columns.
        """

        col_typecode = kwargs.get("typecode", "typecode")
        col_groundspeed = kwargs.get("groundspeed", "groundspeed")
        col_altitude = kwargs.get("altitude", "altitude")
        col_vertical_rate = kwargs.get("vertical_rate", "vertical_rate")
        col_airspeed = kwargs.get("airspeed", "airspeed")
        col_mass = kwargs.get("mass", "mass")
        col_second = kwargs.get("second", None)
        col_d_altitude = kwargs.get("d_altitude", "d_altitude")
        col_d_groundspeed = kwargs.get("d_groundspeed", "d_groundspeed")
        col_d_airspeed = kwargs.get("d_airspeed", "d_airspeed")

        assert col_typecode in flight.columns, f"Column {col_typecode} not found"
        assert col_groundspeed in flight.columns, f"Column {col_groundspeed} not found"
        assert col_altitude in flight.columns, f"Column {col_altitude} not found"
        assert (
            col_vertical_rate in flight.columns
        ), f"Column {col_vertical_rate} not found"

        if col_second is not None:
            assert flight[col_second].dtype.kind in "fiu", (
                "column for second must be float or integer"
            )

        typecode = flight[col_typecode].iloc[0]
        params = self._params_by_type.get(typecode)
        if params is None:
            warnings.warn(f"Aircraft type {typecode!r} not supported")
            result = flight.copy()
            result["fuel_flow"] = np.nan
            result["fuel_flow_kgh"] = np.nan
            if col_second is not None:
                result["fuel_cumsum"] = np.nan
            return result

        dtype = self.dtype
        n = len(flight)
        gs = flight[col_groundspeed].to_numpy(dtype=dtype)
        alt = flight[col_altitude].to_numpy(dtype=dtype)
        vr = flight[col_vertical_rate].to_numpy(dtype=dtype)
        if col_airspeed in flight.columns:
            air = flight[col_airspeed].to_numpy(dtype=dtype)
        else:
            air = gs

        if col_mass in flight.columns:
            mass = flight[col_mass].to_numpy(dtype=dtype)
            mass_norm = (mass - params.OPE_EMPTY_WEIGHT) / (
                params.MAX_TO_WEIGHT - params.OPE_EMPTY_WEIGHT
            )
        else:
            mass_norm = np.full(n, self.DEFAULT_MASS, dtype=dtype)

        if col_second is None:
            dt = None
            d_gs = (
                flight[col_d_groundspeed].to_numpy(dtype=dtype)
                if col_d_groundspeed in flight.columns
                else np.zeros(n, dtype=dtype)
            )
            d_as = (
                flight[col_d_airspeed].to_numpy(dtype=dtype)
                if col_d_airspeed in flight.columns
                else np.zeros(n, dtype=dtype)
            )
            d_alt = (
                flight[col_d_altitude].to_numpy(dtype=dtype)
                if col_d_altitude in flight.columns
                else vr / dtype.type(60.0)
            )
        else:
            sec = flight[col_second].to_numpy(dtype=dtype)
            dt = _diff_bfill(sec)
            d_gs = (
                flight[col_d_groundspeed].to_numpy(dtype=dtype)
                if col_d_groundspeed in flight.columns
                else _diff_bfill(gs) / dt
            )
            d_as = (
                flight[col_d_airspeed].to_numpy(dtype=dtype)
                if col_d_airspeed in flight.columns
                else _diff_bfill(air) / dt
            )
            d_alt = (
                flight[col_d_altitude].to_numpy(dtype=dtype)
                if col_d_altitude in flight.columns
                else _diff_bfill(alt) / dt
            )

        inputs = np.empty((n, 12), dtype=dtype)
        inputs[:, 0] = params.ENGINE_TYPE
        inputs[:, 1] = d_alt
        inputs[:, 2] = d_gs
        inputs[:, 3] = d_as
        inputs[:, 4] = params.SURFACE
        inputs[:, 5] = params.MAX_OPE_ALTI
        inputs[:, 6] = params.MAX_OPE_SPEED
        inputs[:, 7] = alt
        inputs[:, 8] = gs
        inputs[:, 9] = air
        inputs[:, 10] = vr
        inputs[:, 11] = mass_norm

        normalized = (inputs - self.MINIMUMS) / self._scale
        if normalized.dtype != np.float32:
            normalized = normalized.astype(np.float32)

        values = self.session.run(
            [self._output_name], {self._input_name: normalized}
        )[0]
        single = values.squeeze(axis=-1) if values.ndim == 2 else values.ravel()

        scale = float(params.FUEL_FLOW_TO * params.ENGINE_NUM)
        fuel_flow = single * scale

        result = flight.copy()
        result["fuel_flow"] = fuel_flow
        result["fuel_flow_kgh"] = fuel_flow * 3600.0
        if col_second is not None:
            result["fuel_cumsum"] = np.cumsum(fuel_flow * dt)
        return result

    def for_aircraft(self, typecode: str) -> "AircraftFuelEstimator":
        """Return an AircraftFuelEstimator bound to `typecode`, sharing this estimator's
        already-loaded ONNX session and aircraft_params (no reload)."""
        return AircraftFuelEstimator._from_shared(self, typecode)


class AircraftFuelEstimator:
    """Typecode-bound fuel-flow estimator with numpy-only I/O.

    Faster than `FuelEstimator.estimate()` per call because the typecode params
    lookup, normalization arrays, and engine-scale factor are precomputed at
    construction. Input is numpy 1D arrays; output is a numpy 1D array of fuel
    flow in **kg/s** (multiply by 3600 for kg/h).
    """

    DEFAULT_MASS = FuelEstimator.DEFAULT_MASS

    def __init__(
        self,
        typecode: str,
        aircraft_params_path: str = None,
        model_path: str = None,
        dtype=np.float64,
    ):
        """
        Args:
            typecode (str): ICAO aircraft type code (e.g. "A320"). Must exist in
                the aircraft_params table or ValueError is raised.
            aircraft_params_path (str): Path to aircraft params CSV. Default uses package data.
            model_path (str): Path to ONNX model. Default uses package data.
            dtype: Precision for intermediate numpy work (default np.float64).
                The ONNX model still consumes float32; this only governs upstream math.
        """
        if aircraft_params_path is None:
            aircraft_params_path = files("acropole").joinpath(
                "data/aircraft_params.csv"
            )
        params_df = pd.read_csv(aircraft_params_path).set_index("ACFT_ICAO_TYPE")
        if typecode not in params_df.index:
            raise ValueError(f"Aircraft type {typecode!r} not in aircraft_params")

        if model_path is None:
            model_path = files("acropole").joinpath(
                "models/acropole_fuel_model.onnx"
            )
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

        self._init_from(typecode, params_df.loc[typecode], session, np.dtype(dtype))

    @classmethod
    def _from_shared(
        cls, estimator: "FuelEstimator", typecode: str
    ) -> "AircraftFuelEstimator":
        if typecode not in estimator._params_by_type:
            raise ValueError(f"Aircraft type {typecode!r} not in aircraft_params")
        obj = cls.__new__(cls)
        obj._init_from(
            typecode,
            estimator._params_by_type[typecode],
            estimator.session,
            estimator.dtype,
        )
        return obj

    def _init_from(
        self,
        typecode: str,
        params: pd.Series,
        session: ort.InferenceSession,
        dtype: np.dtype,
    ) -> None:
        self.typecode = typecode
        self.dtype = dtype
        self.session = session
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

        cast = dtype.type
        self._engine_type = cast(params.ENGINE_TYPE)
        self._surface = cast(params.SURFACE)
        self._max_ope_alti = cast(params.MAX_OPE_ALTI)
        self._max_ope_speed = cast(params.MAX_OPE_SPEED)
        self._ope_empty_weight = cast(params.OPE_EMPTY_WEIGHT)
        self._mass_range = cast(params.MAX_TO_WEIGHT - params.OPE_EMPTY_WEIGHT)
        self._fuel_scale = float(params.FUEL_FLOW_TO * params.ENGINE_NUM)

        self._mins = np.array(FuelEstimator._MINIMUMS, dtype=dtype)
        self._maxs = np.array(FuelEstimator._MAXIMUMS, dtype=dtype)
        self._scale = self._maxs - self._mins

    def estimate(
        self,
        groundspeed: np.ndarray,
        altitude: np.ndarray,
        vertical_rate: np.ndarray,
        *,
        airspeed: np.ndarray = None,
        mass: np.ndarray = None,
        second: np.ndarray = None,
        d_altitude: np.ndarray = None,
        d_groundspeed: np.ndarray = None,
        d_airspeed: np.ndarray = None,
    ) -> np.ndarray:
        """Predict per-sample fuel flow.

        Required (1D numpy arrays of equal length N):
            groundspeed (knot), altitude (feet), vertical_rate (feet/minute)

        Optional (1D arrays of length N, or None):
            airspeed (knot)            — defaults to groundspeed
            mass (kg)                  — defaults to DEFAULT_MASS sentinel
            second (s)                 — enables Δ/Δt derivatives for any
                                          d_* not explicitly provided
            d_altitude (ft/s)          — pre-computed altitude derivative
            d_groundspeed (kt/s)       — pre-computed groundspeed derivative
            d_airspeed (kt/s)          — pre-computed airspeed derivative

        Returns:
            np.ndarray of shape (N,), fuel flow in kg/s.
        """
        dtype = self.dtype
        gs = np.asarray(groundspeed, dtype=dtype)
        alt = np.asarray(altitude, dtype=dtype)
        vr = np.asarray(vertical_rate, dtype=dtype)
        n = gs.shape[0]

        air = np.asarray(airspeed, dtype=dtype) if airspeed is not None else gs

        if mass is None:
            mass_norm = np.full(n, self.DEFAULT_MASS, dtype=dtype)
        else:
            mass_norm = (
                np.asarray(mass, dtype=dtype) - self._ope_empty_weight
            ) / self._mass_range

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
        else:
            dt = _diff_bfill(np.asarray(second, dtype=dtype))
            d_alt = (
                np.asarray(d_altitude, dtype=dtype)
                if d_altitude is not None
                else _diff_bfill(alt) / dt
            )
            d_gs = (
                np.asarray(d_groundspeed, dtype=dtype)
                if d_groundspeed is not None
                else _diff_bfill(gs) / dt
            )
            d_as = (
                np.asarray(d_airspeed, dtype=dtype)
                if d_airspeed is not None
                else _diff_bfill(air) / dt
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

        values = self.session.run(
            [self._output_name], {self._input_name: normalized}
        )[0]
        single = values.squeeze(axis=-1) if values.ndim == 2 else values.ravel()
        return single * self._fuel_scale
