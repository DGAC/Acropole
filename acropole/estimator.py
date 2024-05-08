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
                "acropole", "models/acropole.keras"
            )

        self.model = tf.keras.models.load_model(model_path)

    def estimate(self, flight: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Estimates fuel flow and consumption based on the flight trajectory.

        Args:
            flight (pd.DataFrame): The flight data as a pandas DataFrame.
            **kwargs: Additional keyword arguments for customization.

        Keyword Args:
            typecode (str): The column name for the aircraft type code. Default is "typecode".
            timestamp (str): The column name for the timestamp. Default is "timestamp".
            groundspeed (str): The column name for the groundspeed. Default is "groundspeed".
            altitude (str): The column name for the altitude. Default is "altitude".
            vertical_rate (str): The column name for the vertical rate. Default is "vertical_rate".
            airspeed (str): The column name for the airspeed. Default is "airspeed".
            mass (str): The column name for the mass. Default is "mass".

        Returns:
            pd.DataFrame: The flight data with additional estimated fuel parameters.

        Raises:
            AssertionError: If the 'timestamp' column is not of type float.

        Warnings:
            If the aircraft type code is not supported.

        Example usage:

            .. code:: python

                import pandas as pd
                from acropole import FuelEstimator

                afe = FuelEstimator()

                flight = pd.DataFrame({
                    "timestamp": [0.0, 1.0, 2.0, 3.0],
                    "typecode": ["A320", "A320", "A320", "A320"],
                    "groundspeed": [400, 410, 420, 430],
                    "altitude": [10000, 11000, 12000, 13000],
                    "vertical_rate": [1000, 1000, 1000, 1000],
                    "airspeed": [400, 410, 420, 430],
                    "mass": [60000, 60000, 60000, 60000]
                })

                flight_fuel = afe.estimate(flight)

        """

        col_typecode = kwargs.get("typecode", "typecode")
        col_timestamp = kwargs.get("timestamp", "timestamp")
        col_groundspeed = kwargs.get("groundspeed", "groundspeed")
        col_altitude = kwargs.get("altitude", "altitude")
        col_vertical_rate = kwargs.get("vertical_rate", "vertical_rate")
        col_airspeed = kwargs.get("airspeed", "airspeed")
        col_mass = kwargs.get("mass", "mass")

        assert (
            flight[col_timestamp].dtype == float
        ), "'timestamp' must be a series of float"

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
        flight = flight.assign(dt=lambda d: d[col_timestamp].diff().bfill()).assign(
            d_altitude=lambda d: (d[col_altitude].diff().bfill() / d.dt),
            d_groundspeed=lambda d: (d[col_groundspeed].diff().bfill() / d.dt),
            d_airspeed=lambda d: (d[col_airspeed].diff().bfill() / d.dt),
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
        data = tf.convert_to_tensor(inputs_normalized)

        single_engine_fuelflow = self.model.predict(data).squeeze()

        flight_fuel = flight_orig.assign(
            fuel_flow=single_engine_fuelflow * flight.ENGINE_NUM,
            fuel_flow_kgh=lambda d: d.fuel_flow * flight.FUEL_FLOW_TO * 3600,
            fuel_cumsum=lambda d: (d.fuel_flow * flight.dt).cumsum(),
        )

        return flight_fuel
