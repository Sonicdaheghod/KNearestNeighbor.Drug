# K Nearest Neighbor - Drug Data Classification
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [Using the Program](#Using-the-Program)

## Purpose of Program
This program was created to understand the basics behind K Nearest Neighbor (KNN) algorithm for data classificaiton. 

## Technologies
Languages/ Technologies used:

* Jupyter Notebook

* Python3

## Setup

Import the following modules and libraries:

``` 
from sklearn import preprocessing
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

```

## Using the program

1. Import and open .csv file of your choice

In this project, the file I used is "drug200.csv". 

```
data = pd.read_csv("drug200.csv",index_col =0)
data.head()

```
<img width="284" alt="image" src="https://github.com/Sonicdaheghod/KNearestNeighbor.Drug/assets/68253811/a96ba117-302e-4c4b-badf-b3ea86dca58d">

2. Since the data in the .csv file is categorical, the machine used here cannot process it. So, we must convert the categorical data to numerical data.

```
integer = preprocessing.LabelEncoder()

Sex = integer.fit_transform(list(data["Sex"]))
BP = integer.fit_transform(list(data["BP"]))
Cholesterol = integer.fit_transform(list(data["Cholesterol"]))
Na_to_K = integer.fit_transform(list(data["Na_to_K"]))
Drug = integer.fit_transform(list(data["Drug"]))
```
if we print out one of the variables after converting the contents, it should print out each data point into a number instead of a string. 

<img width="445" alt="image" src="https://github.com/Sonicdaheghod/KNearestNeighbor.Drug/assets/68253811/a9538199-b4fb-4d4f-88ee-89173bbd0715">

3. After preparing the model for prediction and training the model with the training data, now test the KNN model. 

The model will predict from the data points for the variables "sex","BP", "Na to K" and "Cholesterol" with the corresponding drug that should be administered. We will then compare the prediction with the actual drug (in other words, the correct answer).

<img width="325" alt="image" src="https://github.com/Sonicdaheghod/KNearestNeighbor.Drug/assets/68253811/168847a0-6157-4547-9fb1-09014aad7478">


### Credits
This project was inspired by [Tech with Tim's Tutorial](https://www.techwithtim.net/tutorials/machine-learning-python/k-nearest-neighbors-3)

Dataset from [PRATHAM TRIPATHI](https://www.kaggle.com/datasets/prathamtripathi/drug-classification)
