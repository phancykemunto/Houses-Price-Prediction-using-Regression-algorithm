#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nbconvert


# In[91]:


#importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# # 1.Loading data

# In[92]:


Data=pd.read_csv('C:/PYDATAFILES/HOUSING.csv')
print(Data)


# In[93]:


#Print first few rows of this data
Data.head(10)


# In[94]:


#Extract input (X) and output (y) data from the datase
X = Data.iloc[:, :-1]
Y = Data.iloc[:, -1]
print(X)
print(Y)


# In[6]:


Data.dtypes


# In[7]:


import math
print(math.log(452600))


# In[8]:


Data.columns


# # 2. Handle missing values 

# In[95]:


#Handling missing values
Data.isnull().sum()


# In[96]:


#Fill the missing values with the mean of the total_bedrooms
Data['total_bedrooms'].fillna(value=Data['total_bedrooms'].mean(),inplace= True)
Data.isnull().sum()


# # 3.Encode categorical data :

# In[97]:


ocean_prox = pd.get_dummies(Data['ocean_proximity'],drop_first=False)
print(ocean_prox)


# In[98]:


#Convert categorical column in the dataset to numerical data
le = LabelEncoder()
Data['ocean_proximity']=le.fit_transform(Data['ocean_proximity'])


# In[99]:


train = Data.drop(['longitude', 'latitude'],axis=1)
print(train)


# In[29]:


X = train[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity']]
Y = train['median_house_value']
print(X)
print(Y)


# # 4.Split the dataset

# In[100]:


#80% training dataset and 20% test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=101)
print(X_train, Y_train)
print(X_test, Y_test)


# In[33]:


# Task1.1: Perform Linear Regression on training data

from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_train, Y_train)


# In[35]:


# Task1.2: Predict output for test dataset using the fitted model

predictionLinear = linearRegression.predict(X_test)
print(predictionLinear)


# In[36]:


# Task1.3: Print root mean squared error (RMSE) from Linear Regression

from sklearn.metrics import mean_squared_error
mseLinear = mean_squared_error(Y_test, predictionLinear)
print('Root mean squared error (RMSE) from Linear Regression = ')
print(mseLinear)


# In[37]:


# Task2.1: Perform Decision Tree Regression on training data

from sklearn.tree import DecisionTreeRegressor
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train, Y_train)


# In[39]:


# Task2.2: Predict output for test dataset using the fitted model

predictionDT = DTregressor.predict(X_test)
print(predictionDT)


# In[41]:


# Task2.3: Print root mean squared error from Decision Tree Regression

from sklearn.metrics import mean_squared_error
mseDT = mean_squared_error(Y_test, predictionDT)
print('Root mean squared error from Decision Tree Regression = ')
print(mseDT)


# In[42]:


# Task3.1: Perform Random Forest Regression on training data

from sklearn.ensemble import RandomForestRegressor
RFregressor = RandomForestRegressor()
RFregressor.fit(X_train, Y_train)


# In[44]:


# Task3.2: Predict output for test dataset using the fitted model

predictionRF = RFregressor.predict(X_test)
print(predictionRF)


# In[45]:


# Task3.3: Print root mean squared error from Random Forest Regression

from sklearn.metrics import mean_squared_error
mseRF = mean_squared_error(Y_test, predictionRF)
print('Root mean squared error from Random Forest Regression = ')
print(mseRF)


# In[83]:


X_train_Income=X_train[['median_income']]
X_test_Income=X_test[['median_income']]
print(X_train_Income)
print(X_test_Income)


# In[84]:


print(X_train_Income.shape)
print(Y_train.shape)


# In[86]:


linreg=LinearRegression()
linreg.fit(X_train_Income,Y_train)
y_predict = linreg.predict(X_test_Income)
print(y_predict)


# In[88]:


predictionLinear2 = linreg.predict(X_test_Income)
print(predictionLinear2)


# In[89]:


plt.scatter(X_train_Income, Y_train, color = 'green')
plt.plot (X_train_Income, 
          linreg.predict(X_train_Income), color = 'red')
plt.title ('compare Training result - median_income / median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()


# In[90]:


plt.scatter(X_test_Income, Y_test, color = 'blue')
plt.plot (X_train_Income, 
          linreg.predict(X_train_Income), color = 'red')
plt.title ('compare Testing result - median_income / median_house_value')
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.show()


# In[ ]:




