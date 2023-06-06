#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Credit to:
https://www.techwithtim.net/tutorials/machine-learning-python/k-nearest-neighbors-3
#Dataset from PRATHAM TRIPATHI
#https://www.kaggle.com/datasets/prathamtripathi/drug-classification


# # Import Modules

# In[1]:


from sklearn import preprocessing

import seaborn as sns
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# # Preparing dataset

# In[2]:


data = pd.read_csv("drug200.csv",index_col =0)
data.head()


# # Change values from string to numerical values

# In[10]:


#

integer = preprocessing.LabelEncoder()

Sex = integer.fit_transform(list(data["Sex"]))
BP = integer.fit_transform(list(data["BP"]))
Cholesterol = integer.fit_transform(list(data["Cholesterol"]))
Na_to_K = integer.fit_transform(list(data["Na_to_K"]))
Drug = integer.fit_transform(list(data["Drug"]))

print(Sex)


# In[11]:


#x is for features, y is for labels (target)

x_var = list(zip(Sex,BP,Cholesterol,Na_to_K))
y_var = list(Drug)


# In[14]:


#prepare for prediction
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_var, y_var, test_size = 0.1)


# # Implement K Nearest Neighbor Model

# In[16]:


#Training

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=7)


# In[17]:


model.fit(x_train, y_train)


# In[19]:


#score model, helps us knpw how well this model will predict/ perform with new data
score = model.score(x_test, y_test)
print("Accuracy: ", score)

#accuracy ranges from 0 -1 where 1 = perfect and 0 means no accuracy


# # Test KNN Model

# In[ ]:




