#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Linear regression algorithm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression

# Load the data
oecd_bli = pd.read_csv(r"C:\Users\MAYANK\Downloads\oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(r"C:\Users\MAYANK\Downloads\gdp_per_capita.csv",thousands=',',delimiter='\t',
 encoding='latin1', na_values="n/a")
# Prepare the data
country_stats = pd.merge(oecd_bli, gdp_per_capita,on='Country')
X = np.c_[country_stats["2015"]]
y = np.c_[country_stats["Value"]]
# Visualize the data
country_stats.plot(kind='scatter', x="2015", y='Value')
plt.show()
# Select a linear model
model = sklearn.linear_model.LinearRegression()
# Train the model
model.fit(X, y)
# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]


# In[ ]:




