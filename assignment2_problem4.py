"""
Author: Team 2, STA4241 Summer 2022
Name: assignment2_problem4.py (STA 4241 - Assignment #2B)
Date: 27 May 2022
Description: Problems 4.1-4.5 code, answers, and plot.
May 25 update: Problems 4.1-4.2, 4.4-4.5.
"""

import pandas as pd
from pandas import Series, DataFrame
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Example for cutoff of 0.7
ex_p42 = cmatrix_cutoff(p4data, "HasDetections", "P_HasDetections", cutoff=0.7)
print("--- Confusion Matrix ---")
print("Cutoff probability:\t", ex_p42.cutoff)
print("True positives:\t\t", ex_p42.TP)
print("False positives:\t", ex_p42.FP)
print("True negatives:\t\t", ex_p42.TN)
print("False negatives:\t", ex_p42.FN)
print("Sensitivity:\t\t", ex_p42.sensitivity)
print("Specificity:\t\t", ex_p42.specificity)
print("Accuracy:\t\t\t", ex_p42.accuracy)
print("Precision:\t\t\t", ex_p42.precision)

# Problem 4.3: AUC, Gini

# Problem 4.4
rcurve = {'x':[],'y':[]}
r10curve = pd.DataFrame(rcurve)
for i in range(1,11):
    part=i/10.
    tm = cmatrix_cutoff(p4data, "HasDetections", "P_HasDetections", cutoff=part)
    r10curve=r10curve.append({'x':tm.tpr,'y':tm.fpr}, ignore_index=True)

# Problem 4.5
r5curve = pd.DataFrame(rcurve)
for i in range(1,21):
    part=i/20.
    tm = cmatrix_cutoff(p4data, "HasDetections", "P_HasDetections", cutoff=part)
    r5curve=r5curve.append({'x':tm.tpr,'y':tm.fpr}, ignore_index=True)

plt.plot(r5curve['x'], r5curve['y'])
plt.show()
