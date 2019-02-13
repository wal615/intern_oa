__author__ = 'yshi31'

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm.classes import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from pandas import *

# start and end are positions of testing data
def splitTrainTest(X, y, start, end):
    import scipy.sparse as sp
    if end == X.shape[0]:
        X_train = X[:start]
    elif start == 0:
        X_train = X[end:]
    else:
        X_train = sp.vstack((X[:start], X[end:]), format='csr')
    y_train = y[:start] + y[end:]
    X_test = X[start:end]
    y_test = y[start:end]
    print 'dimension of training matrix : \t', X_train.shape, len(y_train)
    print 'dimension of testing  matrix : \t', X_test.shape, len(y_test)
    return X_train, y_train, X_test, y_test


def eval(y_test, y_pred):
    result = DataFrame({'y_pred' : y_pred,
                        'y_test' : y_test})
    crosstable = crosstab(result['y_pred'], result['y_test'])

    print crosstable

    acc = float(crosstable[1][1]+crosstable[0][0]+crosstable[-1][-1])/len(y_test)
    #negative
    recall_negative = float(crosstable[-1][-1])/(crosstable[-1][-1]+crosstable[0][-1] + crosstable[1][-1])
    prec_negative = float(crosstable[-1][-1])/(crosstable[-1][-1]+crosstable[-1][0]+crosstable[-1][1])
    F1_negative = 2 * prec_negative * recall_negative/( prec_negative + recall_negative)
    #neutral
    recall_neutral = float(crosstable[0][0])/(crosstable[-1][0]+crosstable[0][0] + crosstable[1][0])
    prec_neutral = float(crosstable[0][0])/(crosstable[0][-1]+crosstable[0][0]+crosstable[0][1])
    F1_neutral = 2 * prec_neutral * recall_neutral/(prec_neutral + recall_neutral)
    #positive
    recall_positive = float(crosstable[1][1])/(crosstable[-1][1]+crosstable[0][1] + crosstable[1][1])
    prec_positive = float(crosstable[1][1])/(crosstable[1][-1]+crosstable[1][0]+crosstable[1][1])
    F1_positive = 2 * prec_positive * recall_positive/(prec_positive + recall_positive)

    return acc, prec_negative, recall_negative, F1_negative,recall_neutral,prec_neutral,F1_neutral,recall_positive,prec_positive,F1_positive


def runModel(X, y, model_name):
    nFolders = 10
    accs = []

    precs_negative = []
    recalls_negative = []
    F1s_negative = []

    precs_neutral = []
    recalls_neutral = []
    F1s_neutral = []

    precs_positive = []
    recalls_positive = []
    F1s_positive = []

    n = X.shape[0]
    for exp in range(0, nFolders):
        print '\n\n============================================================================================\nexperiment' , exp
        ### 2.1 split training and testing data
        start = (int)((1-(exp+1) * 1.0/nFolders)*n)
        end = (int)((1-exp * 1.0/nFolders)*n)
        #print n, start, end
        X_train, y_train, X_test, y_test = splitTrainTest(X, y, start, end)
        print 'Running', model_name
        if model_name == 'SVM':
            ### 2.2 build classifier
            # clf = LinearSVC(penalty="l1", dual=False, tol=1e-7)
            clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
            ### 2.3 predict
            y_pred = clf.predict(X_test)
        elif model_name == 'NaiveBayes':
            clf = GaussianNB()
            clf.fit(X_train.todense(), y_train)
            y_pred = clf.predict(X_test.todense())
        elif model_name == 'LogisticRegression':
            clf = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
            clf.fit(X_train.toarray(), y_train)
            y_pred = clf.predict(X_test.toarray())
        else:
            raise Exception("The model name is incorrect!!!")
        ### 2.4 eval
        acc, prec_negative, recall_negative, F1_negative,recall_neutral,prec_neutral,F1_neutral,recall_positive,prec_positive,F1_positive = eval(y_test, y_pred)
        print 'Acc = ', acc
        print 'Precision =', prec_negative,prec_neutral,prec_positive
        print 'Recall=', recall_negative,recall_neutral,recall_positive
        print 'F1 =',  F1_negative,F1_neutral,F1_positive
        accs.append(acc)
        precs_negative.append(prec_negative)
        recalls_negative.append(recall_negative)
        F1s_negative.append(F1_negative)

        precs_neutral.append(prec_neutral)
        recalls_neutral.append(recall_neutral)
        F1s_neutral.append(F1_neutral)

        precs_positive.append(prec_positive)
        recalls_positive.append(recall_positive)
        F1s_positive.append(F1_positive)

    print '\n\n\n'
    print 'avg Acc = ', sum(accs)/len(accs)
    print 'avg Precision = ', sum(precs_negative)/len(precs_negative),sum(precs_neutral)/len(precs_neutral),sum(precs_positive)/len(precs_positive)
    print 'avg Recall = ', sum(recalls_negative)/len(recalls_negative), sum(recalls_neutral)/len(recalls_neutral), sum(recalls_positive)/len(recalls_positive)
    print 'avg F1 = ', sum(F1s_negative)/len(F1s_negative), sum(F1s_neutral)/len(F1s_neutral), sum(F1s_positive)/len(F1s_positive)
    # return sum(accs)/len(accs), sum(precs)/len(precs),  sum(recalls)/len(recalls), sum(F1s)/len(F1s)