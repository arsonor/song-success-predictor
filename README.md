# song-success-predictor
This project implements a machine learning model that predicts if a song has the characteristics of a hit or not.

For this, I managed to reunify several .csv tables that can be downloaded from this link:

https://marianaossilva.github.io/DSW2019/

After preparation and EDA (see [`notebooks`](notebooks/) folder), it resulted in a dataset [`data/dataset_ready.csv`](data/dataset_ready.csv) which comprises 18 columns:
* the **'song_id'** (from Spotify)
* the **'hit' target**: 1 for a hit or 0 if it's not  
The value is derived from a score that take into account the presence in the charts in term of position and longevity
* **16 features**: numerical and categorical like the duration of the song, acoustic features from the song, artist popularity and followers, the genre of the song, etc...

 This is the midterm project as part of the [Machine Learning Zoomcamp](https://github.com/arsonor/machine-learning-zoomcamp).

## Table of contents

[Technologies used in this project](#technologies-used-in-this-project)  
[Installing dependencies](#installing-dependencies)  
[Running and using the application](#running-and-using-the-application)  
[Details on the code](#code)  
[Notebooks](#notebooks)

## Technologies used in this project

- Python 3.12 and scikit-learn library
- Docker for containerization
- Flask as the interface
- Pipenv as the dependency manager
- AWS Elastic Beanstalk for the deployment in the Cloud

## Installing dependencies

Follow these instructions:

1. Run a Codespace from this repository (or fork it, or git clone it, whatever you want to do)
2. Run:

```bash
pip install pipenv
```
3. Install the app dependencies:

```bash
pipenv install --dev
```

## Running and using the application

The easiest way to run the application is with `docker`:

1. Build the image (defined in [Dockerfile](Dockerfile))

```bash
docker build -t hit-prediction .
```

2. Run it:

```bash
docker run -it -p 9696:9696 hit-prediction:latest
```

## Code

The code for the application is the following:

- [`train.py`](train.py) - Train, Validate and Test the final model and save it in a joblib file (the file was too heavy with pickle)
- [`predict.py`](predict.py) - Load the model and serve it via a Flask web service.
- [`test.py`](test.py) - Test the model with a song example.



## Notebooks

For experiments, I use Jupyter notebooks.
They are in the [`notebooks`](notebooks/) folder.

To start Jupyter, run:

```bash
cd notebooks
pipenv run jupyter notebook
```

I have the following notebooks:

- [`data_preparation.ipynb`](notebooks/data_preparation.ipynb): The code for handling songs and merging with different tables
- [`eda.ipynb`](notebooks/eda.ipynb): the Exploratory Data Analysis that resulted in a final dataset
- [`model_classification.ipynb`](notebooks/model_classification.ipynb): Preprocessing, Training and tuning different models for classification task and select the best one

