"""
Author: Team 2, STA4241 Summer 2022
Name: assignment2_problem3.py (STA 4241 - Assignment #2B)
Date: 27 May 2022
Description: Problems 3.1-3.6 code, answers, and plot.
"""

import pandas as pd
from pandas import Series, DataFrame
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
matplotlib.rcParams["figure.dpi"] = 150

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

# Problem 3.6
# Find residuals and add residuals field to dataframe
problem3data['resids'] = problem3data['P_SalePrice'] - problem3data['SalePrice']
xres = problem3data['SalePrice']
yres = problem3data['resids']
# Render residuals plot
plt.style.use("seaborn-darkgrid")
plt.scatter(x=xres, y=yres, marker="o", s=10, color="#08787f", zorder=2, label="")
plt.title("SalePrice Residuals")
plt.xlabel("SalePrice")
plt.ylabel("Residuals")
plt.axhline(y=0, color="#ee2244", linestyle="--", alpha=0.7, linewidth=0.9)
plt.xticks(rotation=45)
# Make loess line
yloess = lowess(yres, xres, frac=0.6/3.0)
plt.plot(yloess[ :,0], yloess[ :,1], color="#ff5b00", label="Loess", linewidth=2, alpha=0.9)
plt.legend(loc="upper right")
plt.show()
