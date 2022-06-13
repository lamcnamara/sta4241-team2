"""
Author: Team 2, STA4241 Summer 2022
Name: assignment5_problem1.py (STA 4241 - Assignment #5)
Date: 16 June 2022
Description: Problems 1 code, answers, and plot.
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sklearn.metrics
import sklearn.utils
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

# Enable for reproducible results. Set to 1000
np.random.seed(1000)

# Problem 1
filename = "C:/Users/zjant/Downloads/ASS_05_Data.csv"
housedata = pd.read_csv(filename)

# Sample with replacement n = 1460, 20 times, generating 20 samples
samples = {}
for k in range(1,21):
    samples[k] = sklearn.utils.resample(housedata, replace=True, n_samples=1460)

# Generate the 20 models from each of the 20 samples
models = {}
Xs = Ys = {}
for k in range(1,21):
    # Predictors
    Xs[k] = pd.DataFrame({'LotArea': samples[k]['LotArea'],
                          'TotalBsmtSF': samples[k]['TotalBsmtSF'],
                          'GarageCars': samples[k]['GarageCars'],
                          'AGE': samples[k]['AGE'],
                          'TotalArea': samples[k]['TotalArea']})
    # Continuous response variable
    Ys[k] = pd.DataFrame({'SalePrice': samples[k]['SalePrice']})
    
    # sklearn decision tree regression, parameters are complete guesses - crossvalidate to find optimal params
    models[k] = DecisionTreeRegressor(criterion="squared_error", max_depth=7, min_samples_split=25)
    models[k].fit(Xs[k], Ys[k])

    #for index, row in samples[k].iterrows():

        #this_xs = pd.DataFrame([row['LotArea'], row['TotalBsmtSF'], row['GarageCars'], row['AGE'], row['TotalArea']])
        #this_array = np.array([row['LotArea'], row['TotalBsmtSF'], row['GarageCars'], row['AGE'], row['TotalArea']])
        #row['Pred_SalePrice'] = models[k].predict(this_xs)
        #row['Tree_Score'] = models[k].score(this_array, row['SalePrice'])
        # Trying to find out how to use predict() here
