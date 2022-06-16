"""
Author: Team 2, STA4241 Summer 2022
Name: assignment5_problem1.py (STA 4241 - Assignment #5)
Date: 17 June 2022
Description: Problem 1 code, answers, and plot.
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
plt.rcParams["figure.autolayout"] = True
import sklearn.metrics
import sklearn.utils
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

np.random.seed(1000) # Enable for reproducible results. Set to 1000

filename = "C:/Users/zjant/Downloads/ASS_05_Data.csv"
housedata = pd.read_csv(filename)
Xall = pd.DataFrame({'LotArea': housedata['LotArea'], 'TotalBsmtSF': housedata['TotalBsmtSF'], 'GarageCars': housedata['GarageCars'], 'AGE': housedata['AGE'], 'TotalArea': housedata['TotalArea']})
Yall = pd.DataFrame({'SalePrice': housedata['SalePrice']})

# Step 01
# Sample with replacement n = 1460, 20 times, generating 20 samples
samples = {}
for k in range(1,21):
    samples[k] = sklearn.utils.resample(housedata, replace=True, n_samples=1460)

# Generate the 20 models from each of the 20 samples, with predictions and model scores
models = {}
Xs = {}
Ys = {}
predicted = {}
scores = {}
for k in range(1,21):
    # Predictors
    Xs[k] = pd.DataFrame({'LotArea': samples[k]['LotArea'],
                          'TotalBsmtSF': samples[k]['TotalBsmtSF'],
                          'GarageCars': samples[k]['GarageCars'],
                          'AGE': samples[k]['AGE'],
                          'TotalArea': samples[k]['TotalArea']})
    # Continuous response variable
    Ys[k] = pd.DataFrame({'SalePrice': samples[k]['SalePrice']})
    # Step 02
    models[k] = DecisionTreeRegressor(criterion="squared_error", max_depth=6, min_samples_split=50) # Experiment with?
    models[k].fit(Xs[k], Ys[k])
    # Step 03
    predicted[k] = models[k].predict(Xall)
    scores[k] = models[k].score(Xall, Yall)

# Step 04
# Ensemble each of 20 model predictions, for each of 1460 observations, to get 1 bagging estimator for each of 1460 obs
baggingest = []
for i, row in housedata.iterrows():
    total = 0
    for k in predicted:
        total += predicted[k][i]
    baggingest.append(total/20)

# Step 05
bagerror = 0
for k in predicted:
    for i, row in housedata.iterrows():
        bagerror += ((predicted[k][i] - baggingest[i])**2)
bagmse = bagerror/20
print("Bagging error estimator:", bagmse)

# Box plot for all 20 predictions + bagging ensemble
predicted['ens'] = np.array(baggingest)
#predicted['obs'] = np.array(housedata['SalePrice'])
preds = pd.DataFrame(predicted)
plt.style.use('seaborn-darkgrid')
ax = preds.plot(kind="box")
plt.title("20 trees and bagging ensemble")
plt.xlabel("Estimator")
plt.ylabel("HousePrice (predicted)")
plt.show()
