# Acropole <img src="https://github.com/DGAC/Acropole/blob/main/logo.png" width="30">

This repository contains the Acropole model for aircraft fuel flow prediction and Python packages for aircraft trajectory processing and fuel flow enhancement.

## Easy Install

For a trouble-free installation, creating a dedicated anaconda environment is recommended :

```sh
conda create -n acropole python=3.12 -c conda-forge
```

Activate the conda environment :

```sh
conda activate acropole
```

Install this library:

```sh
git clone https://github.com/DGAC/Acropole.git
cd Acropole
pip install .

```

## Example of use

Here is a minimal working example:

```python
import pandas as pd
from acropole import FuelEstimator

fe = FuelEstimator()

flight = pd.DataFrame({
    "typecode": ["A320", "A320", "A320", "A320"],
    "groundspeed": [400, 410, 420, 430],
    "altitude": [10000, 11000, 12000, 13000],
    "vertical_rate": [2000, 1500, 1000, 500],

    # optional features:
    "second": [0.0, 1.0, 2.0, 3.0],
    "airspeed": [400, 410, 420, 430],
    "mass": [60000, 60000, 60000, 60000]
})

flight_fuel = fe.estimate(flight)
```

Note:

- When the `second` column is provided, the fuel estimation is more accurate,
  especially due to **derivatives of speeds** (acceleration) used in the estimation.
- `airspeed` is optional. If not provided, it is assumed to be equal
  to groundspeed. However, accurate airspeed is recommended for better estimation.
- Expected sampling rate is 4 seconds, higher or lower sampling rate might induce noisier fuel flow. Resampling data before estimating fuel flow is recommanded.

For a more complete example, refer to `examples/fuel_estimation.ipynb`

## Aircraft data and estimation models

Aircraft parameters from open data to feed the model are available in `data/aircraft_params.csv` and loaded by default. Model data is available in `models/` and also loaded by default.

You can specify your own data and model file with the following initialization of `FuelEstimator`. You need to make sure the same column names are in your aircraft CSV file.

```python
fe = FuelEstimator(
    aircraft_params_path="path/to/your/data.csv",
    model_path="path/to/your/SavedModel/",
)
```

## Model training and evaluation

The Acropole model is a neural network built using data from Quick Access Recorder (QAR) from different aircraft types. Evaluation of the model and list of aircraft is available at https://github.com/DGAC/Acropole/tree/main/evaluation/Dense_Acropole_FuelFlow_Scaling.


## Comparison of Different Model Performances

Comparison of different model performances per phase for 1000 test flights of A320-214 aircraft using real mass and true airspeed.

| Phase | Samples \# | Metric | ACROPOLE | OpenAP | OpenAP V2 | BADA  | Poll-Schumann |
|-------|------------|--------|--------------|--------------|------------|---------------|----------------|
|       |            | **MAPE (%)**  | 2.13    | 30.35                     | 8.84       | 6.53                       | 6.85                                       |
| CLIMB | 1,403,850   | **MAE (kg/min)** | 1.66          | 25.81                     | 6.92       | 5.53                       | 5.65                                       |
|       |            | **ME (kg/min)**  | 0.85     | -25.66                    | -2.48      | -5.27                      | -4.62                                      |
||||||||||
|       |            | **MAPE (%)**  | 4.41   | 18.59                     | 10.69      | 7.01                       | 4.84                                       |
| LEVEL | 4,017,801   | **MAE (kg/min)** | 1.82      | 7.82                      | 3.48       | 2.65                       | 2.03                                       |
|       |            | **ME (kg/min)**  | 1.22     | -7.47                     | 2.64       | -1.43                      | -0.73                                      |
||||||||||
|       |            | **MAPE (%)**  | 12.63      | 51.69                     | 32.4       | 21.50                      | 21.55                                      |
| DESCENT| 1,684,117  | **MAE (kg/min)** | 2.71         | 8.62                      | 5.58       | 3.71                       | 4.71                                       |
|       |            | **ME (kg/min)**  | 1.88         | -1.75                     | -1.16      | -0.64                      | -3.67                                      |
||||||||||
|       |            | **MAPE (%)**  | 5.91       | 27.60                     | 14.71      | 9.84                       | 8.61                                       |
| ALL   | 7,105,768   | **MAE (kg/min)** | 1.99        | 11.55                     | 4.58       | 3.44                       | 3.29                                       |
|       |            | **ME (kg/min)**  | 1.30       | -9.92                     | 0.84       | -2.03                      | -2.09                                      |
||||||||||
|       |            | **Processing time (s)** | 3          | 284                      | 255        | 474                        | 28                                         |



## Credits

```bibtex
@misc{jarry_towards_2024,
    title = {Towards aircraft generic {Quick} {Access} {Recorder} fuel flow regression models for {ADS}-{B} data},
    author = {Jarry, Gabriel and Delahaye, Daniel and Hurter, Christophe},
    month = apr,
    year = {2024},
    doi = {10.13140/RG.2.2.23229.27360},
}

```
