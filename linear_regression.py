import sklearn
import tensorflow
import keras
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

train = False # Set to True if you wish to retrain your linear model
num_trains = 30 # number of times model should be trained

data = pd.read_csv("student-mat.csv", sep = ';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
best = 0

if train:
    for i in range(num_trains):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        acc = linear.score(x_test, y_test)

        if acc>best:
            best = acc
            print(acc)
            with open("studentmodel.pickle", "wb") as f:
                pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)
print("Final model accuracy: ", acc)

p = "G2"
style.use("ggplot")
plt.scatter(data[p], data[predict])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
