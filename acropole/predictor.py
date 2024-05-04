# @internal
#  @author: Gabriel JARRY
# @endinternal

import importlib


class Predictor:
    """
    Class to use fuel flow model
    """

    def __init__(
        self,
        default_model="models/acropole_sequential_fuelflow.keras",
        mock_tensorflow=None,
        mock_pkg_resources=None,
        mock_utils=None,
    ):
        """
        Class initilizer that load dependant packages, and inits model cache.

        :param default_model: default model name as a String
        :param mock_tensorflow: mock for tensorflow package (Default is None)
        :param mock_pkg_resources: mock for pkg_resources package (Default is None)
        :param mock_utils: mock for acropole.utils package (Default is None)
        """
        self.model_name = default_model

        self.np = importlib.import_module("numpy")

        self.fuel_max = self.np.array(
            [1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1]
        )
        self.fuel_min = self.np.array(
            [0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0]
        )

        if mock_tensorflow is None:
            self.tf = importlib.import_module("tensorflow")
        else:
            self.tf = mock_tensorflow

        if mock_pkg_resources is None:
            self.pkg_resources = importlib.import_module("pkg_resources")
        else:
            self.pkg_resources = mock_pkg_resources

        if mock_utils is None:
            self.utils = importlib.import_module("acropole.utils")
        else:
            self.utils = mock_utils

        self.cache = self.utils.compute_once(self.load_generic_model)

    def load_generic_model(self, name):
        """
        Function that loads the generic model for fuel scaling:

        :return: The tensorflow model
        """
        try:
            path = self.pkg_resources.resource_filename("acropole", name)
        except FileNotFoundError:
            path = name
        return self.tf.keras.models.load_model(path, compile=False)

    def predict_fuel_generic(self, model_input_values):
        """
        Function that predicts the generic fuel flow from a list of input values

        :param model_input_values: is the array of inputs:
        :return: List of predicted generic fuel flow
        """
        model = self.cache(self.model_name)
        model_input_values = (model_input_values - self.fuel_min) / (
            self.fuel_max - self.fuel_min
        )
        data = self.tf.constant(model_input_values, dtype=self.tf.float32)
        values = model.predict(data)
        return list(values.flatten())
