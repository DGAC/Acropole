# Acropole <img src="https://github.com/DGAC/Acropole/blob/main/logo.png" width="30">

This repository contains the Acropole model for aircraft fuel flow prediction and Python packages for aircraft trajectory processing and fuel flow enhancement.


## Easy Install

To easy install, creating dedicated anaconda environment is recommended :

```sh
conda create -n acropole python=3.8 -c conda-forge
```

Activate conda environment :

```sh
conda activate acropole
```

Install dependencies :

```sh
conda install numpy tensorflow scipy joblib scikit-learn pandas  
```

If you want to add Jupyter notebooks and matplotlib :

```sh
conda install matplotlib jupyter jinja2==3.0.3 
```

Clone repository :

```sh
git clone https://github.com/DGAC/Acropole.git
```

Finally, install lib :


```sh
cd acropole
pip install .
```

## Python packages

The Acropole Python library includes the following packages :

- columns: that contains default data column names and list of columns

- utils: that contains shared functions

- predictor: that contains functions to load and apply Acropole model

- trajectory: that contains trajectory processes and pipelines

## Data and models
### Available data

Aircraft parameters from open data to feed the model are available in https://github.com/DGAC/Acropole/blob/main/acropole/data/acft_params.csv and loaded by the packages.

### Model training and evaluation

The Acropole model is a neural network built using data from Quick Access Recorder (QAR) from several aircraft types. Evaluation of the model is available in https://github.com/DGAC/Acropole/tree/main/evaluation/Dense_Acropole_FuelFlow_Scaling.

Data set is composed of :

- A320-200 - 16453 trajectories

- A330-223 - 186 trajectories

- ATR72-600 - 2605 trajectories

- B737-85P - 8744 trajectories

- B737-8GJ - 2995 trajectories

- B737-8K2 - 21226 trajectories

- CRJ-1000 - 29422 trajectories

- CRJ-700 - 17234 trajectories

- E-170 - 30462 trajectories

- E-190 - 36287 trajectories

## Example of use

For example of use please refer to https://github.com/DGAC/Acropole/blob/main/examples/examples.ipynb


