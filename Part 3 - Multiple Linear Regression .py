
# coding: utf-8

# In[1]:


#import various relevant libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os 


# In[2]:


#prepare the data 
os.chdir(r"C:\Users\Zack\OneDrive\Documents\Part 2 - Regression\Section 5 - Multiple Linear Regression\Multiple_Linear_Regression\Multiple_Linear_Regression")
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encoding categorical data
# encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#wncoding the dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#avoid dummy variable trap
x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =  0.2, random_state = 0)


# In[3]:


#fit model to training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(x_train, y_train)


# In[4]:


#predicting test set results 
y_pred = regressor.predict(x_test)

#this is an "all-in" model that examines each of the independable variables, regardless of their statistical significance to the dependent variable


# In[7]:


#to optimize the model, build it differently using backward elimination (rather than an "all-in" approach)
import statsmodels.formula.api as sm 
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
x_optimal = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()
#x5 had a p value > 0.05, so remove it and repeat it until there is no p value > 0.05
x_optimal = x[:, [0, 1, 2, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()
x_optimal = x[:, [0, 1, 2, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_optimal).fit()
regressor_OLS.summary()

