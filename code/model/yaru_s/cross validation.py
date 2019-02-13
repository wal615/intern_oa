import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from numpy import *
iris = datasets.load_iris()
iris.data.shape, iris.target.shape

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=1)

X_train.shape, y_train.shape

X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

clf.score(X_test, y_test)
# SVC's socre method Returns the mean accuracy on the given test data
# and labels.

b=(clf.predict(X_train)==y_train)
b = b.astype(float)
print clf.score(X_train,y_train), b.sum()/len(b)

# using cross validation
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
scores1 = cross_val_score(clf, iris.data, iris.target, cv=5, scoring= score_new, verbose=10)
# define a new scorer
def score_new (clf_e,x,y):
    b = (clf_e.predict(x) == y)
    b = b.astype(float)
    acc = b.sum()/len(b)+10
    c = 'hello'
    a = {'a':2,'3':4}
    return a
