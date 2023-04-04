# @internal
#  @author: Gabriel JARRY
# @endinternal


from pkg_resources import resource_filename

import tensorflow as tf
from .utils import *
from .variables import FUEL_GENERIC_MIN, FUEL_GENERIC_MAX, MODEL_ACRPL


def load_generic_model(name):
    """
    Function that loads the generic model for fuel scaling:
    :return: The tensorflow model
    """
    try:
        path = resource_filename('acropole', name)
    except FileNotFoundError:
        path = name
    return tf.saved_model.load(path)


model_cache = compute_once(load_generic_model)


def predict_fuel_generic(model_input_values):
    """
    Function that predicts the generic fuel flow from a list of input values
    :param model_input_values: is the array of inputs:
    :return: List of predicted generic fuel flow
    """
    model = model_cache(MODEL_ACRPL)
    model_input_values = (model_input_values - FUEL_GENERIC_MIN) / (FUEL_GENERIC_MAX - FUEL_GENERIC_MIN)
    data = tf.constant(model_input_values, dtype=tf.float32)
    tensorflow_result_dict = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY](data)
    key, values = tensorflow_result_dict.popitem()
    return list(values.numpy().flatten())


