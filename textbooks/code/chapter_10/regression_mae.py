# Cross Validation Regression MAE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from pandas import read_csv

filename = "housing.csv"
names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
dataframe = read_csv(filename, delim_whitespace=True, names=names)
array = dataframe.values
X = array[:, 0:13]
Y = array[:, 13]
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = "neg_mean_absolute_error"
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("MAE: %.3f (%.3f)" % (results.mean(), results.std()))
