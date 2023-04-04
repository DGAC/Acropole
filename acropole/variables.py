# @internal
#  @author: Gabriel JARRY
# @endinternal
import numpy as np


# Variables

FUEL_GENERIC_MAX = np.array([1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1])
FUEL_GENERIC_MIN = np.array([0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0])

DERIV = "DERIV"
SEP = "_"
BFILL = "bfill"
MODEL_ACRPL = "models/Dense_Acropole_FuelFlow_Scaling"
ACFT_ACRPL = "data/acft_params.csv"

DEFAULT_MIN_CONF_IND = 0.8
HOUR = 3600                       # s
DEFAULT_MASS = -1.0               # kg






