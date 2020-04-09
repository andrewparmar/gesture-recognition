# Ridge Regression
from sklearn.linear_model import Ridge
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
model = Ridge()
scoring = "neg_mean_squared_error"
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
