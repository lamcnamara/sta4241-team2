"""
Author: Team 2, STA4241 Summer 2022
Name: assignment2_problem4.py (STA 4241 - Assignment #2B)
Date: 27 May 2022
Description: Problems 4.1-4.5 code, answers, and plot.
May 25 update: Problems 4.1-4.2.
"""

import pandas as pd
from pandas import Series, DataFrame
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150

# Problem 4.1
filename = "C:/Users/zjant/Downloads/Microsoft_Results.csv"
p4data = pd.read_csv(filename)

# Problem 4.2
class Object: pass
def cmatrix_cutoff(df, actual, predictp, cutoff=0.5, places=3):
    mat = Object()
    mat.cutoff = cutoff
    mat.T = mat.Total = df.shape[0]
    mat.P = mat.Pos = df[(df[actual] == 1)].shape[0]
    mat.N = mat.Neg = df[(df[actual] == 0)].shape[0]
    mat.TP = mat.truepositive = df[((df[actual] == 1) & (df[predictp] < cutoff))].shape[0]
    mat.FP = mat.falsepositive = df[((df[actual] == 0) & (df[predictp] < cutoff))].shape[0]
    mat.TN = mat.truenegative = df[((df[actual] == 0) & (df[predictp] >= cutoff))].shape[0]
    mat.FN = mat.falsenegative = df[((df[actual] == 1) & (df[predictp] >= cutoff))].shape[0]
    mat.tpr = mat.recall = mat.sensitivity = round(mat.TP/(mat.TP+mat.FN), places)
    mat.tnr = mat.specificity = round(mat.TN/(mat.TN+mat.FP), places)
    mat.fnr = round(mat.FN/(mat.FN+mat.TP), places)
    mat.fpr = mat.FalseAlarmRate = round((1-mat.tnr), places)
    mat.accuracy = round((mat.TP+mat.TN)/mat.Total, places)
    if (mat.TP+mat.FP) == 0: # Avoid divzero err
        mat.precision = 0
    else:
        mat.precision = round(mat.TP/(mat.TP+mat.FP), places)
    return mat

# 5th percentiles
rcurve = {'x':[],'y':[]}
rocurve = pd.DataFrame(rcurve)
for d in range(1,21):
    cut=d/20.
    tm = cmatrix_cutoff(p4data, "HasDetections", "P_HasDetections", cutoff=cut, places=3)
    rocurve=rocurve.append({'x':tm.tpr,'y':tm.fpr}, ignore_index=True)
plt.plot(rocurve['x'],rocurve['y'])
plt.show()
