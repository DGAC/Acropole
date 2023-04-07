#  @internal
#
#  @author: Gabriel JARRY
#  @endinternal

# Python libraries
import unittest
import sys
from unittest.mock import Mock, patch, MagicMock

from importlib import import_module
from pathlib import Path
import pandas as pd

# Internal Libs
sys.path.insert(0, str(Path(__file__).parent.parent))


class TrajectoryTests(unittest.TestCase):

    __COL_ARPT = "ARPT"
    __COL_PLOT_DATE = "PLOT_DATE"
    __COL_LATITUDE = "LATITUDE"
    __COL_LONGITUDE = "LONGITUDE"
    __COL_SYST_POINT_ID = "SYST_POINT_ID"
    __COL_FLIGHT_TIME = "FLIGHT_TIME"
    __TIME_LAST_PLOT = "TIME_LAST_PLOT"
    __COLS_PROCESS = [__COL_SYST_POINT_ID, "TIME", __TIME_LAST_PLOT, __COL_FLIGHT_TIME, __COL_PLOT_DATE]
    __COLS_KEEP = [__COL_ARPT]
    __COLS_RESAMPLE = [__COL_LATITUDE]


    @classmethod
    def setUpClass(cls):
        mock_scipy = Mock
        mock_interpolate = Mock()
        mock_interp1d = Mock()
        mock_interpolate.interp1d = mock_interp1d
        mock_scipy.interpolate = mock_interpolate

        mock_latitude = Mock()
        mock_latitude.side_effect = [[22, 23, 24]]
        mock_interp1d.side_effect = [mock_latitude]

        mock_predictor = Mock()


        mock_utils = Mock
        mock_utils.moving_average = Mock()

        cls.trajectory = import_module("acropole.trajectory").Trajectory(mock_scipy=mock_scipy,
                                                                         mock_predictor=mock_predictor)

        values = [["LFMN", "6/27/2020 3:51:30 AM +00:00", 30],
                  ["LFMN", "6/27/2020 3:51:24 AM +00:00", 24],
                  ["LFMN", "6/27/2020 3:51:36 AM +00:00", 36]]

        cls.test_resample_df = pd.DataFrame(values, columns=[cls.__COL_ARPT, cls.__COL_PLOT_DATE,
                                                             cls.__COL_LATITUDE])

        values = [["6/27/2020 3:51:30 AM +00:00", 10.0],
                  ["6/27/2020 3:51:24 AM +00:00", 10.0],
                  ["6/27/2020 3:51:36 AM +00:00", 11.0],
                  ["6/27/2020 3:51:36 AM +00:00", 12.0],
                  ["6/27/2020 3:51:36 AM +00:00", 12.0],
                  ["6/27/2020 3:51:36 AM +00:00", 20.0],
                  ["6/27/2020 3:51:36 AM +00:00", 22.0]]

        cls.test_smooth_df = pd.DataFrame(values, columns=[cls.__COL_PLOT_DATE, cls.__COL_LATITUDE])

    def test_trajectory_resample(self):
        resampled_df = self.trajectory.resample(self.test_resample_df, self.__COLS_KEEP,
                                                                     self.__COLS_RESAMPLE, self.__COLS_PROCESS)

        ids = resampled_df[self.__COL_SYST_POINT_ID].values.tolist()
        flight_time = resampled_df[self.__COL_FLIGHT_TIME].values.tolist()
        arpts = resampled_df[self.__COL_ARPT].values.tolist()
        time_last_plot = resampled_df[self.__TIME_LAST_PLOT].values.tolist()
        plot_date = resampled_df[self.__COL_PLOT_DATE].values.tolist()
        latitude = resampled_df[self.__COL_LATITUDE].values.tolist()
        expected_plot_date = ["2020-06-27 03:51:24+00:00", "2020-06-27 03:51:28+00:00", "2020-06-27 03:51:32+00:00"]
        expected_latitude = [22, 23, 24]

        self.assertEqual(len(resampled_df), 3)
        self.assertListEqual(ids, [1, 2, 3])
        self.assertListEqual(flight_time, [0.0, 4.0, 8.0])
        self.assertListEqual(arpts, ["LFMN"] * 3)
        self.assertListEqual(time_last_plot, [0.0, 4.0, 4.0])
        self.assertListEqual(plot_date, expected_plot_date)
        self.assertListEqual(latitude, expected_latitude)

    def test_trajectory_smooth(self):
        smoothed_df = self.trajectory.smooth(self.test_smooth_df,
                                                                  cols_smooth=self.__COLS_RESAMPLE, window_width=4)

        smoothed_latitude = smoothed_df[self.__COL_LATITUDE].values.tolist()
        self.assertListEqual(smoothed_latitude, [10, 10, 11.0, 13.0, 15.4, 20.0, 22.0])


if __name__ == "__main__":
    unittest.main()
