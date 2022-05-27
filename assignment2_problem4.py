"""
Author: Team 2, STA4241 Summer 2022
Name: assignment2_problem4.py (STA 4241 - Assignment #2B)
Date: 27 May 2022
Description: Problems 4.1-4.5 code, answers, and plot.
"""

import pandas as pd
from pandas import Series, DataFrame
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 150
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sklearn.metrics

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
    mat.TP = mat.truepositive = df[((df[actual] == 1) & (df[predictp] > cutoff))].shape[0]
    mat.FP = mat.falsepositive = df[((df[actual] == 0) & (df[predictp] > cutoff))].shape[0]
    mat.TN = mat.truenegative = df[((df[actual] == 0) & (df[predictp] <= cutoff))].shape[0]
    mat.FN = mat.falsenegative = df[((df[actual] == 1) & (df[predictp] <= cutoff))].shape[0]
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

"""Should output:
--- Confusion Matrix ---
Cutoff probability:	 0.7
True positives:		 84108
False positives:	 14959
True negatives:		 486040
False negatives:	 414893
Sensitivity:		 0.169
Specificity:		 0.97
Accuracy:			 0.57
Precision:			 0.849
"""

# Problem 4.3
p4auc = sklearn.metrics.roc_auc_score(p4data["HasDetections"], p4data["P_HasDetections"])
p4gini = 2 * (p4auc - 0.5)
print("AUC:", round(p4auc, 3))
print("Gini coefficient:", round(p4gini, 3))

"""Should output:
AUC: 0.694
Gini coefficient: 0.388
"""

# Problem 4.4
rcurve = {'x':[],'y':[]}
r10curve = pd.DataFrame(rcurve)
for i in range(1,11):
    part=i/10.
    tm = cmatrix_cutoff(p4data, "HasDetections", "P_HasDetections", cutoff=part)
    r10curve=r10curve.append({'x':tm.fpr,'y':tm.tpr}, ignore_index=True)
plt.style.use("seaborn-darkgrid")
plt.plot([0, 1], [0, 1], linestyle="dashed", color="#666666", alpha=0.6, label="Random")
plt.plot(r10curve['x'], r10curve['y'], color="#dd3b00", zorder=2, label="Model")
plt.scatter(r10curve['x'], r10curve['y'], color="#dd3b00", s=25, marker="o", label="", zorder=3)
plt.title("ROC curve, decile level")
plt.xlabel("False alarm rate (1 - Specificity)")
plt.ylabel("Sensitivity (TPR)")
plt.legend(loc="lower right", facecolor="white", framealpha=0.6, frameon=1)
plt.show()

# Problem 4.5
r5curve = pd.DataFrame(rcurve)
for i in range(1,21):
    part=i/20.
    tm = cmatrix_cutoff(p4data, "HasDetections", "P_HasDetections", cutoff=part)
    r5curve=r5curve.append({'x':tm.fpr,'y':tm.tpr}, ignore_index=True)
plt.style.use("seaborn-darkgrid")
plt.plot([0, 1], [0, 1], linestyle="dashed", color="#666666", alpha=0.6, label="Random")
plt.plot(r5curve['x'], r5curve['y'], color="#dd3b00", zorder=2, label="Model")
plt.scatter(r5curve['x'], r5curve['y'], color="#dd3b00", s=25, marker="o", label="", zorder=3)
plt.title("ROC curve, five percent level")
plt.xlabel("False alarm rate (1 - Specificity)")
plt.ylabel("Sensitivity (TPR)")
plt.legend(loc="lower right", facecolor="white", framealpha=0.6, frameon=1)
plt.show()
