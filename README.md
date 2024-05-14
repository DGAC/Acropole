# Acropole <img src="https://github.com/DGAC/Acropole/blob/main/logo.png" width="30">

This repository contains the Acropole model for aircraft fuel flow prediction and Python packages for aircraft trajectory processing and fuel flow enhancement.


## Easy Install

To easy install, creating dedicated anaconda environment is recommended :

```sh
conda create -n acropole python=3.12 -c conda-forge
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

The Acropole model is a neural network built using data from Quick Access Recorder (QAR) from different aircraft types. Evaluation of the model and list of aircraft is available in https://github.com/DGAC/Acropole/tree/main/evaluation/Dense_Acropole_FuelFlow_Scaling.


## Example of use

For example of use please refer to https://github.com/DGAC/Acropole/blob/main/examples/examples.ipynb

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





