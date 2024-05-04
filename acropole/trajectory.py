# @internal
#  @author: Gabriel JARRY
# @endinternal

import importlib
from datetime import timedelta

from .columns import (
    COL_ACFT_ICAO_TYPE,
    COLS_ACFT_PARAMS,
    COLS_FUEL,
    COLS_INPUT_FUEL,
    COLS_KEEP,
    COLS_PROCESS,
    COLS_RESAMPLE,
    COLS_SMOOTH,
)


class Trajectory:
    """
    Class that contains data pipelines for trajectory enhancement
    """

    __4SEC = 4  # s
    __EARTH_RADIUS = 6371.37706  # km
    __NM = 1852  # m
    __KM = 1000  # m
    __HOUR = 3600  # s

    __ARR = "arr"
    __DERIV = "DERIV"
    __SEP = "_"

    __ACFT_DICT_PATH = "data/acft_params.csv"

    __DEFAULT_MASS = -1.0

    def __init__(
        self,
        sample_rate=4.0,
        min_duration_s=20,
        min_conf_ind=0.8,
        mock_scipy=None,
        mock_pkg_resources=None,
        mock_predictor=None,
        mock_utils=None,
    ):
        """
        Class initilizer that load dependant packages, and inits model cache.

        :param sample_rate: Sample rate for resample methods as a Float (default 20s)
        :param min_duration_s: Minimum time duration for segment cco/cdo as a Float (default 20s)
        :param min_conf_ind: Minimum confidence index to compute fuel flow using Acropole Lib (Default is 0.8)
        :param mock_scipy: mock for scipy package (Default is None)
        :param mock_predictor: mock for predictor package (Default is None)
        :param mock_utils: mock for acropole.utils package (Default is None)

        """
        self.sample_rate = sample_rate
        self.min_duration_s = min_duration_s
        self.min_conf_ind = min_conf_ind

        self.np = importlib.import_module("numpy")
        self.pd = importlib.import_module("pandas")

        if mock_scipy is None:
            self.scipy = importlib.import_module("scipy")
        else:
            self.scipy = mock_scipy

        if mock_pkg_resources is None:
            self.pkg_resources = importlib.import_module("pkg_resources")
        else:
            self.pkg_resources = mock_pkg_resources

        if mock_predictor is None:
            self.predictor = importlib.import_module("acropole.predictor").Predictor()
        else:
            self.predictor = mock_predictor

        if mock_utils is None:
            self.utils = importlib.import_module("acropole.utils")
        else:
            self.utils = mock_utils

        self.acft_dict = self.load_generic_acft_dict(self.__ACFT_DICT_PATH)

    def resample(
        self,
        df,
        cols_keep=COLS_KEEP,
        cols_resample=COLS_RESAMPLE,
        cols_process=COLS_PROCESS,
    ):
        """
        Function that resamples trajectory.
        :param df: trajectory as a Pandas Dataframe
        :param cols_keep: Names of columns to keep as a List of String (default COLS_KEEP)
        :param cols_resample: Names of columns to resample as a List of String (default COLS_RESAMPLE)
        :param cols_process: Names of columns for process as a List of String (default COL_PROCESS)
        :return: The resampled trajectory as a Pandas Dataframe
        """
        (
            col_syst_point_id,
            col_time,
            col_time_last_plot,
            col_flight_time,
            col_plot_date,
        ) = cols_process

        df[col_time] = self.pd.to_datetime(df[col_plot_date])
        df = df.sort_values(by=col_time)
        df[col_time_last_plot] = df[col_time] - df[col_time].shift(1)
        df[col_time_last_plot] = (
            df[col_time_last_plot]
            .apply(lambda dt: dt.seconds + dt.microseconds / 10e5)
            .fillna(0.0)
        )
        df[col_flight_time] = df[col_time_last_plot].cumsum()

        start = df[col_flight_time].iloc[0]
        end = df[col_flight_time].iloc[-1]
        r = self.np.arange(start, end, self.sample_rate)
        res_df = self.pd.DataFrame()

        for col in cols_keep:
            res_df[col] = [df[col].iloc[0]] * len(r)

        res_df[col_flight_time] = r
        res_df[col_time_last_plot] = [0.0] + [4.0] * (len(r) - 1)
        res_df[col_syst_point_id] = range(1, len(res_df) + 1)
        res_df[col_plot_date] = df.TIME.iloc[0] + res_df[col_flight_time].apply(
            lambda el: timedelta(seconds=el)
        )
        res_df[col_plot_date] = res_df[col_plot_date].apply(str)

        for col in cols_resample:
            interp = self.scipy.interpolate.interp1d(df[col_flight_time], df[col])
            res_df[col] = interp(r)

        return res_df

    def smooth(self, df, cols_smooth=COLS_SMOOTH, window_width=4):
        """
        Function that smooths given columns of a trajectory using moving average smoothing.
        :param df: trajectory as a Pandas Dataframe
        :param cols_smooth: Names of columns to smooth as a List of String (default COLS_SMOOTH)
        :param window_width: window width as an Integer  (default 4)
        :return: The current trajectory with columns smoothed as a Pandas Dataframe
        """
        for col in cols_smooth:
            df.loc[:, col] = self.utils.moving_average(df[col], window_width)
        return df

    def load_generic_acft_dict(self, name):
        """
        Function that loads the aircracft parameters dict
        :return: The aircracft parameters dict
        """
        try:
            path = self.pkg_resources.resource_filename("acropole", name)
        except FileNotFoundError:
            path = name
        acft_df = self.pd.read_csv(path, sep=";")
        acft_dict = {row[COL_ACFT_ICAO_TYPE]: row for key, row in acft_df.iterrows()}
        return acft_dict

    def fuel_prediction(
        self, df, acft_dict, cols_fuel=COLS_FUEL, mass_kg=None, tas=None
    ):
        """
        Function that computes derivates and predict fuel consumption over trajectory using Acropole Librairy
        :param df: trajectory as a Pandas Dataframe
        :param acft_dict: Dictionary of acft parameter as a Dict : String -> Pd.Series(params)
        :param cols_fuel: Names of columns for fuel process as a List of String (default COLS_FUEL)
        :param mass_kg: Mass column name or None if no mass as a String
        :param tas: True Air Speed column name or None if no mass as a String
        :return: The current trajectory with extra fuel information as a Pandas Dataframe
        """
        (
            col_estim_fuel_flow_acrpl,
            col_estim_conso_kg,
            col_estim_fuel_flow_kgh,
            col_grnd_spd_kt,
            col_true_air_spd_kt,
            col_mass,
            col_flpl_airc_type,
            col_engine_num,
            col_fuel_flow_to,
            col_conf_ind,
            col_deriv_tas_kt,
            col_deriv_gs_kt,
            col_time_last_plot,
            col_alti_std_ft,
            col_oew,
            col_mtow,
        ) = cols_fuel

        acft_type = df[col_flpl_airc_type].iloc[0]
        if tas:
            df.loc[:, col_true_air_spd_kt] = df.loc[:, tas]
        else:
            df.loc[:, col_true_air_spd_kt] = df[col_grnd_spd_kt]

        for col in [col_alti_std_ft, col_grnd_spd_kt, col_true_air_spd_kt]:
            df.loc[:, self.__DERIV + self.__SEP + col] = (
                df[col] - df[col].shift(1)
            ).bfill() / df[col_time_last_plot]

        if acft_type in acft_dict.keys():
            params = acft_dict[acft_type]

            for col in COLS_ACFT_PARAMS[1:]:
                df.loc[:, col] = params[col]

            if mass_kg:
                df.loc[:, col_mass] = (df.loc[:, mass_kg] - df[col_oew]) / (
                    df[col_mtow] - df[col_oew]
                )
            else:
                df.loc[:, col_mass] = self.__DEFAULT_MASS

            input_values = df[COLS_INPUT_FUEL]

            if params[col_conf_ind] > self.min_conf_ind:
                df.loc[:, col_estim_fuel_flow_acrpl] = (
                    self.predictor.predict_fuel_generic(input_values)
                )
            else:
                df.loc[:, col_estim_fuel_flow_acrpl] = None

            df.loc[:, col_estim_fuel_flow_kgh] = (
                df[col_estim_fuel_flow_acrpl]
                * df[col_engine_num]
                * df[col_fuel_flow_to]
                * self.__HOUR
            )
            df.loc[:, col_estim_conso_kg] = (
                df[col_estim_fuel_flow_kgh] * df[col_time_last_plot] / self.__HOUR
            )
        else:
            for col in COLS_ACFT_PARAMS[1:]:
                df.loc[:, col] = None

            df.loc[:, col_estim_fuel_flow_acrpl] = None
            df.loc[:, col_estim_fuel_flow_kgh] = None
            df.loc[:, col_estim_conso_kg] = None
        return df

    def trajectory_process(
        self,
        df,
        cols_keep=COLS_KEEP,
        cols_resample=COLS_RESAMPLE,
        cols_smooth=COLS_SMOOTH,
        mass_kg=None,
        tas=None,
        apply_smoothing=False,
    ):
        """
        Function that preprocess trajectory Dataframe with resampling, moving average smoothing, adding distance and
        fuel along trajectory.
        :param df: Trajectory as a Pandas Dataframe
        :param cols_keep: Names of columns to keep as a List of String (default COLS_KEEP)
        :param cols_resample: Names of columns to resample as a List of String (default COLS_RESAMPLE)
        :param cols_smooth: Names of columns to smooth as a List of String (default COLS_SMOOTH)
        :param mass_kg: Mass column name or None if no mass as a String (default None)
        :param tas: True Air Speed column name or None if no mass as a String (default None)
        :param apply_smoothing: Boolean if true smoothing is applied (default False)
        :return: The preprocessed trajectory as a Pandas Dataframe
        """
        df = self.resample(df, cols_keep=cols_keep, cols_resample=cols_resample)
        if apply_smoothing:
            df = self.smooth(df, cols_smooth=cols_smooth)
        df = self.fuel_prediction(df, self.acft_dict, mass_kg=mass_kg, tas=tas)
        return df
