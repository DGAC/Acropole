# @internal
#  @author: Gabriel JARRY
# @endinternal

import numpy as np


def moving_average(values_list, window_width):
    """
    Function that smooths given values using moving average.
    :param values_list: Values to smooth as a List of (float or int)
    :param window_width: window width as an Integer
    :return: The smoothed values as a List of (float or int)
    """
    result_list = []
    for i, t in enumerate(values_list):
        if i < window_width // 2 or i > len(values_list) - 1 - window_width // 2:
            result_list.append(values_list[i])
        else:
            sub_list = np.array([values_list[j]
                                 for j in range(max(0, i - window_width // 2),
                                                min(i + window_width // 2 + 1, len(values_list)))])
            result_list.append((sum(sub_list) / (len(sub_list))).tolist())
    return result_list


def compute_once(function_to_memoize):
    """
    Return a memoïzation of a function that store the result at first apply and the use the stored value
    :param function_to_memoize: is a function to memoïze
    :return: he memoïzed function
    """
    cache = {}

    def wrapper(param):
        if param not in cache:
            cache[param] = function_to_memoize(param)
        return cache[param]

    return wrapper





