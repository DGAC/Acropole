#  @internal
#  @author: Gabriel JARRY
#  @endinternal

# Python libraries
import unittest
import sys

from importlib import import_module
from unittest.mock import patch, Mock
from pathlib import Path
import numpy as np

# Internal Libs
sys.path.insert(0, str(Path(__file__).parent.parent))


class UtilsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.utils = import_module("acropole.utils")

        cls.moving_average_list = [10.0, 10.0, 11.0, 12.0, 12.0, 20.0, 22.0]
        cls.moving_average_result = [10, 10, 11.0, 13.0, 15.4, 20.0, 22.0]

    def test_moving_average(self):
        moveing_average_result = self.utils.moving_average(self.moving_average_list, 4)
        for v1, v2 in zip(moveing_average_result, self.moving_average_result):
            self.assertEqual(v1, v2)

    def test_compute_once(self):
        mock_compute_once = Mock(return_value="Resultat")
        memoised_function = self.utils.compute_once(mock_compute_once)
        result = [memoised_function("A320") for _ in range(10)]
        self.assertListEqual(["Resultat"] * 10, result)
        self.assertEqual(mock_compute_once.call_count, 1)


if __name__ == "__main__":
    unittest.main()
