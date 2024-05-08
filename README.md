# Acropole <img src="https://github.com/DGAC/Acropole/blob/main/logo.png" width="30">

This repository contains the Acropole model for aircraft fuel flow prediction and Python packages for aircraft trajectory processing and fuel flow enhancement.

## Easy Install

For a trouble-free installation, create a dedicated anaconda environment is recommended :

```sh
conda create -n acropole python=3.11 -c conda-forge
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
	"timestamp": [0.0, 1.0, 2.0, 3.0],
	"typecode": ["A320", "A320", "A320", "A320"],
	"groundspeed": [400, 410, 420, 430],
	"altitude": [10000, 11000, 12000, 13000],
	"vertical_rate": [2000, 1500, 1000, 500],

	# airspeed and mass are optional
	# "airspeed": [400, 410, 420, 430],
	# "mass": [60000, 60000, 60000, 60000]
})

flight_fuel = fe.estimate(flight)
```

For a more complete example, refer to `examples/example.ipynb`

## Aircraft data and estimation models

Aircraft parameters from open data to feed the model are available in `data/aircraft_params.csv` and loaded by default. Model data is available in `models/` and also loaded by default.

You can specify your own data and model file with the following initialization of `FuelEstimator`. You need to make sure the same column names are in your aircraft csv file.

```python
fe = FuelEstimator(
	aircraft_params_path="path/to/your/data.csv",
	model_path="path/to/your/model.kears",
)
```

## Model training and evaluation

The Acropole model is a neural network built using data from Quick Access Recorder (QAR) from different aircraft types. Evaluation of the model and list of aircraft is available in https://github.com/DGAC/Acropole/tree/main/evaluation/Dense_Acropole_FuelFlow_Scaling.

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
