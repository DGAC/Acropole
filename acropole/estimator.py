import warnings

import numpy as np
import pandas as pd
import pkg_resources
import tensorflow as tf


class FuelEstimator:
    """
    Class that contains data pipelines for trajectory enhancement
    """

    DEFAULT_MASS = -1.0
    MAXIMUMS = np.array([1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1])
    MINIMUMS = np.array([0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0])

    def __init__(self, aircraft_params_path: str = None, model_path: str = None):
        """
        Initializes the Trajectory class.

        Args:
            aircraft_table_path (str): The path to the aircraft table. Default is None (use package data).
            model_path (str): The path to the prediction model. Default is None (use package data).

        """

        if aircraft_params_path is None:
            aircraft_params_path = pkg_resources.resource_filename(
                "acropole", "data/aircraft_params.csv"
            )

        self.aircraft_params = pd.read_csv(aircraft_params_path)

        if model_path is None:
            model_path = pkg_resources.resource_filename(
                "acropole", "models/acropole_fuel_model"
            )

        model = tf.saved_model.load(model_path)
        self.predictor = model.signatures[
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]

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
            - timestamp (str): The column name for the timestamp (in second).
                Default is "timestamp".
            - airspeed (str): The column name for the airspeed (in knot).
                Default is "airspeed".
            - mass (str): The column name for the mass (in kilogram).
                Default is "mass".

        Returns:
            pd.DataFrame: The flight data with additional estimated fuel parameters.

        Note:
            - When `timestamp` is provided, the fuel estimation is more accurate,
                especially due to **derivatives of speeds** (acceleration) used in the estimation.
            - `airspeed` is optional. If not provided, it is assumed to be equal
                to groundspeed. However, accurate airspeed is recommended for better estimation.


        Warnings:
            If the aircraft type code is not supported.

        Example usage:

            .. code:: python

                import pandas as pd
                from acropole import FuelEstimator

                afe = FuelEstimator()

                flight = pd.DataFrame({
                    "typecode": ["A320", "A320", "A320", "A320"],
                    "groundspeed": [400, 410, 420, 430],
                    "altitude": [10000, 11000, 12000, 13000],
                    "vertical_rate": [1000, 1000, 1000, 1000],

                    # optional features:
                    # "timestamp": [0.0, 1.0, 2.0, 3.0],
                    # "airspeed": [400, 410, 420, 430],
                    # "mass": [60000, 60000, 60000, 60000]
                })

                flight_fuel = afe.estimate(flight)

        """

        col_typecode = kwargs.get("typecode", "typecode")
        col_groundspeed = kwargs.get("groundspeed", "groundspeed")
        col_altitude = kwargs.get("altitude", "altitude")
        col_vertical_rate = kwargs.get("vertical_rate", "vertical_rate")
        col_airspeed = kwargs.get("airspeed", "airspeed")
        col_mass = kwargs.get("mass", "mass")

        assert col_typecode in flight.columns, f"Column {col_typecode} not found"
        assert col_groundspeed in flight.columns, f"Column {col_groundspeed} not found"
        assert col_altitude in flight.columns, f"Column {col_altitude} not found"
        assert (
            col_vertical_rate in flight.columns
        ), f"Column {col_vertical_rate} not found"

        col_second = kwargs.get("second", None)
        if col_second is not None:
            assert (
                flight[col_second].dtype == float or int
            ), "column for second must be float or integer"

        flight_typecode = flight[col_typecode].iloc[0]
        if flight_typecode not in self.aircraft_params.ACFT_ICAO_TYPE.unique():
            warnings.warn(
                f"Aircraft type {flight_typecode} flight_typecode not supported"
            )

        flight_orig = flight.copy()

        flight = flight.merge(
            self.aircraft_params,
            how="left",
            left_on=col_typecode,
            right_on="ACFT_ICAO_TYPE",
        )

        if col_airspeed not in flight.columns:
            flight = flight.assign(airspeed=lambda d: d[col_groundspeed])

        if col_mass not in flight.columns:
            flight = flight.assign(mass_norm=self.DEFAULT_MASS)
        else:
            flight = flight.assign(
                mass_norm=lambda d: (d[col_mass] - d.OPE_EMPTY_WEIGHT)
                / (d.MAX_TO_WEIGHT - d.OPE_EMPTY_WEIGHT)
            )

        # compute devrivatives of altitude and speeds
        if col_second is None:
            # if no timestamp provided, assume quasi-steady state
            flight = flight.assign(
                d_groundspeed=0,
                d_airspeed=0,
                d_altitude=lambda d: d[col_vertical_rate] / 60,  # ft/s
            )
        else:
            flight = flight.assign(dt=lambda d: d[col_second].diff().bfill()).assign(
                d_groundspeed=lambda d: (d[col_groundspeed].diff().bfill() / d.dt),
                d_airspeed=lambda d: (d[col_airspeed].diff().bfill() / d.dt),
                d_altitude=lambda d: (d[col_altitude].diff().bfill() / d.dt),
            )

        inputs = flight[
            [
                "ENGINE_TYPE",
                "d_altitude",
                "d_groundspeed",
                "d_airspeed",
                "SURFACE",
                "MAX_OPE_ALTI",
                "MAX_OPE_SPEED",
                col_altitude,
                col_groundspeed,
                col_airspeed,
                col_vertical_rate,
                "mass_norm",
            ]
        ]

        inputs_normalized = (inputs - self.MINIMUMS) / (self.MAXIMUMS - self.MINIMUMS)
        data = tf.convert_to_tensor(inputs_normalized, dtype=tf.float32)

        key, values = self.predictor(data).popitem()
        single_engine_fuelflow = values.numpy().squeeze()

        flight_fuel = flight_orig.assign(
            fuel_flow=single_engine_fuelflow * flight.FUEL_FLOW_TO * flight.ENGINE_NUM,
            fuel_flow_kgh=lambda d: d.fuel_flow * 3600,
        )

        if col_second is not None:
            flight_fuel = flight_fuel.assign(
                fuel_cumsum=lambda d: (d.fuel_flow * flight.dt).cumsum()
            )

        return flight_fuel
