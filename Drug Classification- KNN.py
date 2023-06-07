#!/usr/bin/env python
# coding: utf-8

# In[107]:


#Credit to:
#https://www.techwithtim.net/tutorials/machine-learning-python/k-nearest-neighbors-3
#Dataset from PRATHAM TRIPATHI
#https://www.kaggle.com/datasets/prathamtripathi/drug-classification


# # Import Modules

# In[108]:


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

# In[109]:


data = pd.read_csv("drug200.csv",index_col =0)
data.head()


# # Change values from string to numerical values

# In[110]:


#

integer = preprocessing.LabelEncoder()

Sex = integer.fit_transform(list(data["Sex"]))
BP = integer.fit_transform(list(data["BP"]))
Cholesterol = integer.fit_transform(list(data["Cholesterol"]))
Na_to_K = integer.fit_transform(list(data["Na_to_K"]))
Drug = integer.fit_transform(list(data["Drug"]))

print(Sex)


# In[111]:


#x is for features, y is for labels (target)

x_var = list(zip(Sex,BP,Cholesterol,Na_to_K))
y_var = list(Drug)


# In[112]:


#prepare for prediction
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_var, y_var, test_size = 0.1)


# # Implement K Nearest Neighbor (KNN) Model

# In[113]:


#Training

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)


# In[114]:


model.fit(x_train, y_train)


# In[115]:


#score model, helps us knpw how well this model will predict/ perform with new data
score = model.score(x_test, y_test)
print("Accuracy: ", score)

#accuracy ranges from 0 -1 where 1 = perfect and 0 means no accuracy


# # Test KNN Model

# ### This allows us to compare how the model performs on the individual points of data in our test data. Compares the predicted drug  vs the actual drug based on the given variable points.

# In[116]:


my_predicted = model.predict(x_test)
the_names = ["DrugA","DrugB","DrugC","DrugX","DrugY"]

#iteration through each data point
for x in range(len(my_predicted)):
    print("Predicted: ", the_names[my_predicted[x]], "Data: ", x_test[x], "Actual: ", the_names[y_test[x]])



# # Determining Neighboring Points

# ### In KNN, the model determines the target of a data point depending on the target label on the nearby points, also known as the neighboring points.

# In[119]:


my_predicted = model.predict(x_test)
the_names = ["DrugA","DrugB","DrugC","DrugX","DrugY"]

#iteration through each data point
for x in range(len(my_predicted)):
    print("Predicted: ", the_names[my_predicted[x]], "Data: ", x_test[x], "Actual: ", the_names[y_test[x]])
    
    #identifying what neighboring points used for each data point
    neighboring_points = model.kneighbors([x_test[x]], 5, True)
    print("The neighboring points used: ", neighboring_points)

