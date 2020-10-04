# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:33:14 2020

@author: NSK
"""

# Multiple Linear Regresssion
# importing libraries


import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


# importing dataset
dataset = pd.read_csv(r"C:\Users\yash\Desktop\Machine learning\My PRograms\Python\Regression\2 Multiple Linear Regression\50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])], remainder = "passthrough")
x = ct.fit_transform(x)

# Avoiding the dummy variable trap
#x = x[:, 1:]

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

# training the model training set.
from sklearn.linear_model import LinearRegression
MLR=LinearRegression()
MLR.fit(x_train,y_train)

# Predicting the results on the test set 
pred = MLR.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((pred.reshape(len(pred),1),y_test.reshape(len(y_test),1)),1))
