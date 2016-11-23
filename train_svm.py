import os
import pdb
import numpy as np
import caffe, h5py
import glob as glob
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

def same_person_classifier(data_train,label_train,data_test,label_test,search= False):
    svr = svm.SVC()
    parameter = {'kernel':('linear', 'rbf'), 'C':[0.1,1, 10,100]}
    cv_clf = grid_search.GridSearchCV(svr,parameter)
    cv_clf.fit(data_train,label_train)
    test_accuray = clf.score(data_test,label_test)
    print "cv_clf.best_params_: ",cv_clf.best_params_
    clf =cv_clf
    Acc_train  = clf.score(data_train, label_train)
    Acc_test  = clf.score(data_test, label_test)
	
    return clf, Acc_train,Acc_test


if __name__ =='__main__':
    f_train = h5py.File('train_hard.h5','r')
    train_data = np.asarray(f_train['data'])
    train_label = np.asarray(f_train['label']).argmax(axis=1)
    label_train = train_label[0:train_data.shape[0]:5]
    data_train = train_data[0:train_data.shape[0]:5,:]
    f_test = h5py.File('test_hard.h5','r')
    data_test = np.asarray(f_test['data'])
    label_test = np.asarray(f_test['label']).argmax(axis=1)
	
    clf,Acc_train,Acc_test = same_person_classifier(data_train,label_train,data_test,label_test,search=True)
		
    print "train_acc: " + str(Acc_train) + " test_acc: " + str(Acc_test)
    pkl.dump(clf,open('svm_classifier','w'))

