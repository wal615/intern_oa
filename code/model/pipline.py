__author__ = 'yshi31'
sys.path.append("/Users/Ben/PycharmProjects/cs583/project2/model")
from classifier import *
from data import *

obama_tw = tweetdata('Obama','Obama.csv')
print obama_tw.data.shape
X,y = obama_tw.vectorize()
runModel(X, y, 'LogisticRegression')

romney_tw = tweetdata('Romney','Romney.csv')
print romney_tw.data.shape
X,y = romney_tw.vectorize()
runModel(X, y, 'LogisticRegression')