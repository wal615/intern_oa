__author__ = 'Xuelong Wang'

from sklearn.svm.classes import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import *
from numpy import *
from pandas import *


def upsample_training(X):
    # Indicies of each class' observations
    y = X['apply']
    i_class0 = np.where(y == 0)[0]
    i_class1 = np.where(y == 1)[0]

    # Number of observations in each class
    n_class0 = len(i_class0)
    n_class1 = len(i_class1)

    # For every observation in class 1, randomly sample from class 0 with
    # replacement
    i_class1_upsampled = np.random.choice(i_class1, size=n_class0, replace=True)

    # get the final large index with its order shuffled 
    index_upsample = np.concatenate([i_class1_upsampled, i_class0])
    index_upsample = np.random.choice(index_upsample, size=len(index_upsample),
    replace=False)

    X_balance = X.iloc[index_upsample,:]
    return(X_balance)

def split_training(X, fold, state):

    kf = KFold(n_splits=fold, shuffle=True, random_state=state)
    split_data = kf.split(X)
    return split_data, fold


def runModel(X, y, S_data, model_name):

    f = open('r_'+"_"+model_name+'.txt', 'w')
    auc_score_all = []
    fold = S_data[1]
    Index_gen = S_data[0]
    label = ['0.0', '1.0']
    # note that python does not copy the generator,
    # so when it's in the end of the for loop the generator for S_data is also extruded!
    print 'Running', model_name
    for exp in range(0, fold):
        print "="*80,"\n", "experiment =", exp
        # getting one fold of indices from the index generator
        train_index, test_index = Index_gen.next()
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # fitting the model
        # 1 fit a model
        # 2 prediction
        if model_name == 'SVM':
            # LinearSVC take care of the multi class response by using one vs others method
            clf = LinearSVC(random_state=0).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        elif model_name == 'NaiveBayes':
            clf = GaussianNB()
            clf.fit(X_train.to_dense(), y_train)
            y_pred = clf.predict(X_test.to_dense())
        elif model_name == 'LogisticRegression':
            clf = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
            clf.fit(X_train.as_matrix(), y_train)
            y_pred = clf.predict(X_test.as_matrix())
        else:
            raise Exception("The model name is incorrect!!!")
        ### 2.4 eval 
        auc_score = roc_auc_score(y_test, y_pred, average=None)
        auc_score_all.append(auc_score)

    auc_ave = mean(array(auc_score_all), 0)

    print >> f, model_name, '\n',"="*80
    print >> f, 'avg auc = ', auc_ave
