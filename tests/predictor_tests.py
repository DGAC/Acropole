#  @internal
#  @author: Gabriel JARRY
#  @endinternal

# Python libraries
import unittest
import sys
from unittest.mock import Mock

from importlib import import_module
from pathlib import Path

# Internal Libs
sys.path.insert(0, str(Path(__file__).parent.parent))


class PredictorTests(unittest.TestCase):

    __COL_POINT_ID = "SYST_POINT_ID"

    __LOAD_GENERIC_MODEL_VALUES = [("acropole", "Dense_Acropole_FuelFlow_Scaling")]

    @classmethod
    def setUpClass(cls):
        cls.mock_tensorflow = Mock()
        cls.mock_load = Mock()
        cls.mock_tensorflow.saved_model.load = cls.mock_load

        cls.mock_pkg_resources = Mock()
        cls.mock_resource_filename = Mock()
        cls.mock_pkg_resources.resource_filename = cls.mock_resource_filename

        cls.mock_utils = Mock()
        mock_cache = Mock(return_value=Mock(return_value=1.0))
        cls.mock_utils.compute_once = Mock(return_value=mock_cache)

        cls.predictor = import_module("acropole.predictor").Predictor(
            mock_tensorflow=cls.mock_tensorflow,
            mock_pkg_resources=cls.mock_pkg_resources,
            mock_utils=cls.mock_utils
        )

    def test_predictor_load_generic_model(self):

        self.predictor.load_generic_model("Dense_Acropole_FuelFlow_Scaling")
        self.assertEqual(self.mock_resource_filename.call_count, 1)
        self.assertEqual(self.mock_load.call_count, 1)
        args, kwargs = self.mock_resource_filename.call_args_list[0]
        self.assertEqual(args, self.__LOAD_GENERIC_MODEL_VALUES[0])


if __name__ == "__main__":
    unittest.main()
