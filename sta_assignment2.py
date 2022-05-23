"""
Author: Team 2, STA4241 Summer 2022
Name: sta_assignment2.py (STA 4241 - Assignment #2B)
Date: 27 May 2022
Description: Model performance functions, problems 3.1-3.6 and 4.1-4.5.
22 May update: contains results for problem 1.1 and code for 3.1-3.5.
"""

import pandas as pd
from pandas import Series, DataFrame

# Problem 3.1
# Import CSV (when it's uploaded)
# filename = "C:/Users/zjant/Documents/as2a-problem1-1-model1.csv"
# data1 = pd.read_csv(filename)

# Problem #3.2
# Find residual sum of squares
# Takes dataframe, observations field and predictions field
def sse(df, obs, pred):
    ssr = 0
    for index, row in df.iterrows():
        ssr += (row[pred] - row[obs])**2
    return ssr

# Find average squared error
# Takes dataframe, observations field and predictions field
def ase(df, obs, pred):
    ave = sse(df, obs, pred)/df.shape[0] # .shape[0] is row count in pandas
    return ave

# Problem 3.3
# Find coefficient of determination (R^2)
# Takes dataframe, observations field and predictions field
def rsquared(df, obs, pred):
    obsmean = df[obs].mean()
    num = denom = 0
    for index, row in df.iterrows():
        num += (row[obs] - row[pred])**2
        denom += (row[obs] - obsmean)**2
    r2 = 1 - (num/denom)
    return r2

# Problem 3.4
# Find mean absolute percentage error
# Takes dataframe, observations field and predictions field
def mape(df, obs, pred):
    total = 0
    for index, row in df.iterrows():
        total += abs((row[obs] - row[pred])/row[obs])
    pcerr = total/df.shape[0]
    return pcerr

# Problem 3.5
# Find mean absolute error
# Takes dataframe, observations field and predictions field
def mae(df, obs, pred):
    total = 0
    for index, row in df.iterrows():
        total += abs(row[pred] - row[obs])
    abserr = total/df.shape[0]
    return abserr

# Problem 1.1 data
problem11data = {'response': [3, 4, 5, 6, 7], 
                 'm1pred': [3.2, 4.3, 4.9, 5.7, 6.9], 
                 'm2pred': [3.3, 4.2, 4.8, 5.9, 7.1]}
problem11 = pd.DataFrame(problem11data)

# Problem 1.1-1
print("SSE of model 1:", round(sse(problem11, "response", "m1pred"), 4))
print("SSE of model 2:", round(sse(problem11, "response", "m2pred"), 4))

# Problem 1.1-2
print("ASE of model 1:", round(ase(problem11, "response", "m1pred"), 4))
print("ASE of model 2:", round(ase(problem11, "response", "m2pred"), 4))

# Problem 1.1-3
print("R2 of model 1:", rsquared(problem11, "response", "m1pred"))
print("R2 of model 2:", rsquared(problem11, "response", "m2pred"))

# Problem 1.1-4
print("MAPE of model 1:", round(mape(problem11, "response", "m1pred"), 4)*100, "%")
print("MAPE of model 2:", round(mape(problem11, "response", "m2pred"), 4)*100, "%")

# Problem 1.1-5
print("MAE of model 1:", round(mae(problem11, "response", "m1pred"), 4))
print("MAE of model 2:", round(mae(problem11, "response", "m2pred"), 4))

"""
# TODOs
# Problem 3.6
def plot_residuals_loess():
    return

# Problem 4.1
# Import CSV

# Problem 4.2
def confusion_matrix():
    return

# Problem 4.3
def auc():
    return

def gini():
    return

# Problem 4.4
def roc_decile():
    return
# roc_decile()

# Problem 4.5
def roc_5ile():
    return
# roc_5ile()

def ks():
    return
"""