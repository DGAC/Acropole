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


class PredictorTests(unittest.TestCase):

    __LOAD_GENERIC_MODEL_VALUES = [("acropole", "models/Dense_Acropole_FuelFlow_Scaling")]

    @classmethod
    def setUpClass(cls):
        cls.tensorflow_mock = Mock()
        sys.modules['tensorflow'] = cls.tensorflow_mock

        cls.predictor = import_module("acropole.predictor")
        cls.variables = import_module("acropole.variables")


    @patch("acropole.predictor.tf.saved_model.load")
    @patch("acropole.predictor.resource_filename")
    def test_predictor_load_generic_model(self, ressource_filename_mock, load_model_mock):
        models_mock = MagicMock()
        load_model_mock.side_effect = models_mock

        self.predictor.load_generic_model("models/Dense_Acropole_FuelFlow_Scaling")
        self.assertEqual(ressource_filename_mock.call_count, 1)
        self.assertEqual(load_model_mock.call_count, 1)
        args, kwargs = ressource_filename_mock.call_args_list[0]
        self.assertEqual(args, self.__LOAD_GENERIC_MODEL_VALUES[0])


if __name__ == "__main__":
    unittest.main()
