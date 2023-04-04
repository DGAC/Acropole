from setuptools import setup, find_packages
from pathlib import Path

THIS_DIR = Path(__file__).parent

with THIS_DIR.joinpath("README.md").open("r") as readme_file:
    long_description = readme_file.read()

with THIS_DIR.joinpath("requirements.txt").open("r") as requirements_file:
    requirements = requirements_file.read()

setup(
    name='acropole',
    version='1.0.0',
    description="Open Source lib to predict aircraft fuel flow from radar data using machine learning models",
    author='Gabriel JARRY',
    author_email='gabriel.jarry@aviation-civile.gouv.fr',
    license="DSNA",
    long_description=long_description,
    packages=find_packages(),
    url="",
    install_requires=[requirements],
    package_data={'acropole': ['*'],
                  'acropole.evaluation': ['*'],
                  'acropole.data': ['*'],
                  'acropole.models': ['*'],
                  'acropole.models.Dense_Acropole_FuelFlow_Scaling': ['*'],
                  'acropole.models.Dense_Acropole_FuelFlow_Scaling.assets': ['*'],
                  'acropole.models.Dense_Acropole_FuelFlow_Scaling.variables': ['*'],
                  }
)


