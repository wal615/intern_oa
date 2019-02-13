__author__ = 'Xuelong Wang'

from sklearn.svm.classes import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import *
from numpy import *
from pandas import *


def split_training(X, fold, state):

    kf = KFold(n_splits=fold, shuffle=True, random_state=state)
    split_data = kf.split(X)
    return split_data, fold


def runModel(X, y, S_data, model_name, presedent):

    f = open('r_'+presedent+"_"+model_name+'.txt', 'w')
    accs = []
    p_all = []
    r_all = []
    f1_all = []
    fold = S_data[1]
    Index_gen = S_data[0]
    label = ['-1.0', '0.0', '1.0']
    # note that python does not copy the generator,
    # so when it's in the end of the for loop the generator for S_data is also extruded!
    print 'Running', model_name, presedent
    for exp in range(0, fold):
        print "="*80,"\n", "experiment =", exp
        # getting one fold of indices from the index generator
        train_index, test_index = Index_gen.next()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fitting the model
        # 1 fit a model
        # 2 prediction
        if model_name == 'SVM':
            # LinearSVC take care of the multi class response by using one vs others method
            clf = LinearSVC(random_state=0).fit(X_train, y_train)
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

        p = precision_score(y_test.astype(str), y_pred.astype(str), labels=label, average=None)
        r = recall_score(y_test.astype(str), y_pred.astype(str), labels=label, average=None)
        f1 = f1_score(y_test.astype(str), y_pred.astype(str), labels=label, average=None)
        # overall accuracy for each group
        a = accuracy_score(y_test.astype(str), y_pred.astype(str))
        class_report = classification_report(y_test.astype(str), y_pred.astype(str), labels=label, digits=3)
        print (class_report)
        accs.append(a)
        p_all.append(p)
        r_all.append(r)
        f1_all.append(f1)

    p_ave = mean(array(p_all), 0)
    r_ave = mean(array(r_all), 0)
    f1_ave = mean(array(f1_all), 0)

    print >> f, presedent+"_"+model_name, '\n',"="*80
    print >> f, 'avg Acc = ', sum(accs)/len(accs)
    print >> f, 'avg Precision = ', p_ave
    print >> f, 'avg Recall = ', r_ave
    print >> f, 'avg F1 = ', f1_ave
    # return {"Ave": [p_ave, r_ave, f1_ave], "All": [p_all, r_all, f1_all]}
