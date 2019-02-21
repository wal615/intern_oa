__author__ = 'Xuelong'
import sys
sys.path.append("/home/xuelong/dev/intern_oa/code/model/")
from data_classdoor import *
from classifier_glassdoor import *

apply_data  = glassdoor('Apply_Rate_2019_id_freq.csv')
apply_data.scale_data() # scale covariates and remove NA's 
train_data, test_data = apply_data.split_data()

# using up-sampling method to handel the unbalance class problem
train_data = upsample_training(train_data)

## training 
# split into X and Y
X_train = train_data.iloc[:,0:-3]
Y_train = train_data['apply'] 
X_test = test_data.iloc[:,0:-3]
Y_test = test_data['apply'] 

# cv-traning
#S_data = split_training(X_train, fold=3, state=1)
#runModel(X_train, Y_train, S_data, 'SVM')

#S_data = split_training(X_train, fold=3, state=1)
#runModel(X_train, Y_train, S_data, 'NaiveBayes')

#S_data = split_training(X_train, fold=3, state=1)
#runModel(X_train, Y_train, S_data, 'LogisticRegression')

## testing
# fitting model on the training dataset 
clf = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
clf.fit(X_train.as_matrix(), Y_train)

from IPython import embed; embed()
Y_pred = clf.predict(X_test.as_matrix())
auc_score = roc_auc_score(Y_test, Y_pred, average=None)
print auc_score

