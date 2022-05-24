"""
Author: Team 2, STA4241 Summer 2022
Name: sta_assignment2.py (STA 4241 - Assignment #2B)
Date: 27 May 2022
Description: Model performance functions, problems 3.1-3.6 and 4.1-4.5.
24 May update: contains code for 3.1-3.5.
Maintained by Lauren McNamara (mcnamaralaurenh@gmail.com).
"""

import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

# Problem 3.1
filename = "C:/Users/zjant/Downloads/House_Prices_PRED.csv"
problem3data = pd.read_csv(filename)

# Problem #3.2
def sse(df, obs, pred):
    """Returns residual sum of squares (sum of squared error)."""
    ssr = 0
    for index, row in df.iterrows():
        ssr += (row[pred] - row[obs])**2
    return ssr

def ase(df, obs, pred):
    """Returns average squared error."""
    ave = sse(df, obs, pred)/df.shape[0] # .shape[0] is row count
    return ave

# Problem 3.3
def rsquared(df, obs, pred):
    """Returns coefficient of determination R^2 (R-squared)."""
    obsmean = df[obs].mean()
    num = denom = 0
    for index, row in df.iterrows():
        num += (row[obs] - row[pred])**2
        denom += (row[obs] - obsmean)**2
    r2 = 1 - (num/denom)
    return r2

# Problem 3.4
def mape(df, obs, pred):
    """Returns mean absolute percentage error between 0 and 1.
    Remember to multiply by 100 for percent!"""
    total = 0
    for index, row in df.iterrows():
        total += abs((row[obs] - row[pred])/row[obs])
    pcerr = total/df.shape[0]
    return pcerr

# Problem 3.5
def mae(df, obs, pred):
    """Returns mean absolute error."""
    total = 0
    for index, row in df.iterrows():
        total += abs(row[pred] - row[obs])
    abserr = total/df.shape[0]
    return abserr

# 3.2 answer
print("SSE:", round(sse(problem3data, "SalePrice", "P_SalePrice"), 4))
print("ASE:", round(ase(problem3data, "SalePrice", "P_SalePrice"), 4))
# 3.3 answer
print("R^2:", round(rsquared(problem3data, "SalePrice", "P_SalePrice"), 4))
# 3.4 answer
print("MAPE:", round(mape(problem3data, "SalePrice", "P_SalePrice"), 4) * 100, "%")
# 3.5 answer
print("MAE:", round(mae(problem3data, "SalePrice", "P_SalePrice"), 4))

"""Should output as follows:
SSE: 740014639177.1655
ASE: 506859341.9022
R^2: 0.9196
MAPE: 7.03 %
MAE: 12470.8337
"""


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