# @internal
#  Created on 24 February 2023
#
#  @author: Gabriel JARRY
# @endinternal

from .columns import __COLS_SMOOTH, __COLS_KEEP, __COLS_RESAMPLE, __COLS_ACFT_PARAMS, __COLS_PROCESS, \
    __COLS_INPUT_FUEL, __COLS_FUEL, __COL_ACFT_ICAO_TYPE
from .variables import __4SEC, __DERIV, __SEP, __DEFAULT_MASS, __HOUR, __BFILL, \
    __DEFAULT_MIN_CONF_IND, __ACFT_ACRPL

from .utils import moving_average, compute_once
from .predictor import predict_fuel_generic

from pkg_resources import resource_filename
import numpy as np
import pandas as pd
import scipy
from datetime import timedelta


def resample(df, cols_keep=__COLS_KEEP, cols_resample=__COLS_RESAMPLE, cols_process=__COLS_PROCESS):
    """
    Function that resamples trajectory.
    :param df: trajectory as a Pandas Dataframe
    :param cols_keep: Names of columns to keep as a List of String (default __COLS_KEEP)
    :param cols_resample: Names of columns to resample as a List of String (default __COLS_RESAMPLE)
    :param cols_process: Names of columns for process as a List of String (default __COL_PROCESS)
    :return: The resampled trajectory as a Pandas Dataframe
    """
    col_syst_point_id, col_time, col_time_last_plot, col_flight_time, col_plot_date = cols_process
    df[col_time] = pd.to_datetime(df[col_plot_date], infer_datetime_format=True)
    df = df.sort_values(by=col_time)
    df[col_time_last_plot] = df[col_time] - df[col_time].shift(1)
    df[col_time_last_plot] = df[col_time_last_plot].apply(lambda dt: dt.seconds + dt.microseconds / 10e5).\
        fillna(0.0)
    df[col_flight_time] = df[col_time_last_plot].cumsum()

    start = df[col_flight_time].iloc[0]
    end = df[col_flight_time].iloc[-1]
    r = np.arange(start, end, __4SEC)
    res_df = pd.DataFrame()

    for col in cols_keep:
        res_df[col] = [df[col].iloc[0]] * len(r)

    res_df[col_flight_time] = r
    res_df[col_time_last_plot] = [0.0] + [4.0] * (len(r)-1)
    res_df[col_syst_point_id] = range(1, len(res_df)+1)
    res_df[col_plot_date] = df.TIME.iloc[0] + res_df[col_flight_time].apply(lambda el: timedelta(seconds=el))
    res_df[col_plot_date] = res_df[col_plot_date].apply(str)

    for col in cols_resample:
        interp = scipy.interpolate.interp1d(df[col_flight_time], df[col])
        res_df[col] = interp(r)

    return res_df


def smooth(df, cols_smooth=__COLS_SMOOTH, window_width=4):
    """
    Function that smooths given columns of a trajectory using moving average smoothing.
    :param df: trajectory as a Pandas Dataframe
    :param cols_smooth: Names of columns to smooth as a List of String (default __COLS_SMOOTH)
    :param window_width: window width as an Integer  (default 4)
    :return: The current trajectory with columns smoothed as a Pandas Dataframe
    """
    for col in cols_smooth:
        df.loc[:, col] = moving_average(df[col], window_width)
    return df


def fuel_prediction(df, acft_dict, min_conf_ind=__DEFAULT_MIN_CONF_IND, cols_fuel=__COLS_FUEL,
                    mass_kg=None, tas=None):
    """
    Function that computes derivates and predict fuel consumption over trajectory using Acropole Librairy
    :param df: trajectory as a Pandas Dataframe
    :param acft_dict: Dictionary of acft parameter as a Dict : String -> Pd.Series(params)
    :param min_conf_ind: Minimum confidence index to compute fuel flow using Acropole Lib
    :param cols_fuel: Names of columns for fuel process as a List of String (default __COLS_FUEL)
    :param mass_kg: Mass column name or None if no mass as a String
    :param tas: True Air Speed column name or None if no mass as a String
    :return: The current trajectory with extra fuel information as a Pandas Dataframe
    """
    col_estim_fuel_flow_acrpl, col_estim_conso_kg, col_estim_fuel_flow_kgh,  col_grnd_spd_kt, col_true_air_spd_kt, \
        col_mass, col_flpl_airc_type, col_engine_num, col_fuel_flow_to, col_conf_ind,  col_deriv_tas_kt, \
        col_deriv_gs_kt, col_time_last_plot, col_alti_std_ft, col_oew, col_mtow = cols_fuel
    acft_type = df[col_flpl_airc_type].iloc[0]
    if tas:
        df.loc[:, col_true_air_spd_kt] = df.loc[:, tas]
    else:
        df.loc[:, col_true_air_spd_kt] = df[col_grnd_spd_kt]

    for col in [col_alti_std_ft, col_grnd_spd_kt, col_true_air_spd_kt]:
        df.loc[:, __DERIV + __SEP + col] = (df[col] - df[col].shift(1)).fillna(method=__BFILL) / __4SEC

    if acft_type in acft_dict.keys():
        params = acft_dict[acft_type]

        for col in __COLS_ACFT_PARAMS[1:]:
            df.loc[:, col] = params[col]

        if mass_kg:
            df.loc[:, col_mass] = (df.loc[:, mass_kg] - df[col_oew]) / (df[col_mtow] - df[col_oew])
        else:
            df.loc[:, col_mass] = __DEFAULT_MASS

        input_values = df[__COLS_INPUT_FUEL]

        if params[col_conf_ind] > min_conf_ind:
            df.loc[:,  col_estim_fuel_flow_acrpl] = predict_fuel_generic(input_values)
        else:
            df.loc[:,  col_estim_fuel_flow_acrpl] = None

        df.loc[:,  col_estim_fuel_flow_kgh] = (df[col_estim_fuel_flow_acrpl] * df[col_engine_num] *
                                               df[col_fuel_flow_to] * __HOUR)
        df.loc[:,  col_estim_conso_kg] = df[col_estim_fuel_flow_kgh] * df[col_time_last_plot] / __HOUR
    else:
        for col in __COLS_ACFT_PARAMS[1:]:
            df.loc[:, col] = None

        df.loc[:,  col_estim_fuel_flow_acrpl] = None
        df.loc[:,  col_estim_fuel_flow_kgh] = None
        df.loc[:,  col_estim_conso_kg] = None
    return df


def load_generic_acft_dict(name):
    """
    Function that loads the aircracft parameters dict
    :return: The aircracft parameters dict
    """
    try:
        path = resource_filename('acropole', name)
    except FileNotFoundError:
        path = name
    acft_df = pd.read_csv(path, sep=";")
    acft_dict = {row[__COL_ACFT_ICAO_TYPE]: row for key, row in acft_df.iterrows()}
    return acft_dict


acft_cache = compute_once(load_generic_acft_dict)


def trajectory_process(df, cols_keep=__COLS_KEEP, cols_resample=__COLS_RESAMPLE, cols_smooth=__COLS_SMOOTH,
                       acft_dict=acft_cache(__ACFT_ACRPL), mass_kg=None, tas=None, apply_smoothing=False):
    """
    Function that preprocess trajectory Dataframe with resampling, moving average smoothing, adding distance and fuel
    along trajectory.
    :param df: Trajectory as a Pandas Dataframe
    :param cols_keep: Names of columns to keep as a List of String (default __COLS_KEEP)
    :param cols_resample: Names of columns to resample as a List of String (default __COLS_RESAMPLE)
    :param cols_smooth: Names of columns to smooth as a List of String (default __COLS_SMOOTH)
    :param acft_dict: Dictionary of acft parameter as a Dict : String -> Pd.Series(params) (Default is acft_params)
    :param mass_kg: Mass column name or None if no mass as a String (default None)
    :param tas: True Air Speed column name or None if no mass as a String (default None)
    :param apply_smoothing: Boolean if true smoothing is applied (default False)
    :return: The preprocessed trajectory as a Pandas Dataframe
    """
    df = resample(df, cols_keep=cols_keep, cols_resample=cols_resample)
    if apply_smoothing:
        df = smooth(df, cols_smooth=cols_smooth)
    df = fuel_prediction(df, acft_dict, mass_kg=mass_kg, tas=tas)
    return df
