# K Nearest Neighbor - Drug Data Classification
by Megan Tran

## Table of Contents
* [Purpose of Program](#Purpose-of-program)
* [Technologies](#technologies)
* [Setup](#setup)
* [How to Use the Program](#How-to-Use-the-Program)

## Purpose of Program
This program was created to understand the basics behind K Nearest Neighbor (KNN) algorithm for data classificaiton. 

## Technologies
Languages/ Technologies used:

*Jupyter Notebook
*Python3

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


### Credits
This project was inspired by [Tech with Tim's Tutorial](https://www.techwithtim.net/tutorials/machine-learning-python/k-nearest-neighbors-3)
Dataset from [PRATHAM TRIPATHI](https://www.kaggle.com/datasets/prathamtripathi/drug-classification)
