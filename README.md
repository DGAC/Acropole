# Acropole ![logo](https://gitlab.asap.dsna.fr/dsna/me/acropole/-/blob/main/logo.png)


This repository contains the Acropole models for aircraft fuel flow prediction and Python packages for aircraft trajectory processing and fuel flow enhancement.


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
conda install tensorflow scipy joblib scikit-learn pandas matplotlib jupyter jinja2==3.0.3 
```

Clone repository :

```sh
git clone https://gitlab.asap.dsna.fr/dsna/me/acropole.git
```

Finally, install lib :


```sh
cd acropole
pip install .
```


