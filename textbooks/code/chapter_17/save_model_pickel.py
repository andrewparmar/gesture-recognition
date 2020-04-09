# Save Model Using Pickle
from pickle import dump, load

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pandas import read_csv

filename = "pima-indians-diabetes.data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=7
)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = "finalized_model.sav"
dump(model, open(filename, "wb"))

# some time later...

# load the model from disk
loaded_model = load(open(filename, "rb"))
result = loaded_model.score(X_test, Y_test)
print(result)
