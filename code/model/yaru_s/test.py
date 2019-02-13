import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from numpy import *
from sklearn.metrics import *
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
    a = (acc, c)
    return a

# divide your data into k-fold and each time different parts as the testing part
import numpy as np
from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=3)
for train, test in kf.split(X):
    print("%s %s" % (train, test))


# index array
b = X[0,].toarray()
c = where (b>0)
b[c[0],c[1]]


# Naive Bayes

import numpy as np
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X, Y)

print(clf.predict(X[2:3]))


import numpy as np
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
c_m = clf.fit(X, y)

print(clf.predict(X[2:3]))




# Confusion matrix

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)

y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])

confusion_matrix(y_test, y_pred)

# classification report
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
##############################
# crosstab
target_names = ['class -1', 'class 0', 'class 1']
cr = classification_report(y_test, y_pred, target_names=target_names, digits=3)
result = DataFrame({'y_pred' : y_pred,
                    'y_test' : y_test})
crosstable = crosstab(result['y_pred'], result['y_test'])

# negative
recall_negative = float(crosstable[-1][-1]) / (crosstable[-1][-1] + crosstable[0][-1] + crosstable[1][-1])
prec_negative = float(crosstable[-1][-1]) / (crosstable[-1][-1] + crosstable[-1][0] + crosstable[-1][1])
F1_negative = 2 * prec_negative * recall_negative / (prec_negative + recall_negative)

# built-in socre functions
f1_score(y_test, y_pred, average= None)

# how to print
from sklearn.metrics import classification_report
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_pred, target_names=target_names))
c = classification_report(y_true, y_pred, target_names=target_names)
f = open('result.txt', 'w')
print >> f, 'Filename:', c  # or f.write('...\n')
f.write(c)
f.close()

f = open("result_"+a+".txt", 'w')
f = open('result_1.txt', 'w')

