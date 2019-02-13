__author__ = 'Xuelong'
import sys
sys.path.append("/Users/Ben/PycharmProjects/cs583/project2/model")
from classifier_1 import *
from data_1 import *

obama_tw = tweetdata('Obama','Obama_1.csv')
print "clean data size", obama_tw.data.shape
X, y = obama_tw.vectorize()

S_data = split_training(X, fold=10, state=1)
runModel(X, y, S_data, 'SVM', obama_tw.president)

S_data = split_training(X, fold=10, state=1)
runModel(X, y, S_data, 'NaiveBayes', obama_tw.president)

S_data = split_training(X, fold=10, state=1)
runModel(X, y, S_data, 'LogisticRegression', obama_tw.president)

###############################################################

Romney_tw = tweetdata('Romney', 'Romney_1.csv')
print "clean data size", Romney_tw.data.shape
X, y = Romney_tw.vectorize()

S_data = split_training(X, fold=10, state=1)
runModel(X, y, S_data, 'SVM', Romney_tw.president)

S_data = split_training(X, fold=10, state=1)
runModel(X, y, S_data, 'NaiveBayes', Romney_tw.president)

S_data = split_training(X, fold=10, state=1)
runModel(X, y, S_data, 'LogisticRegression', Romney_tw.president)
