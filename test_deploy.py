import re
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
import pdb
import pickle as pkl
import h5py
from sklearn import preprocessing

#np.random.seed(1)
#"""constructing unbiassed testing set """
#count = 0
#file1 = pkl.load(open('../feature_VGG_test_fc6','rb'))
#X = np.matrix(file1['data'])
#Y = np.array(file1['label'])
#X = preprocessing.normalize(X,norm='l2').T
#shuffle_label = np.unique(Y)
#np.random.shuffle(shuffle_label)
#sample_num =10000
#
#data = np.zeros([sample_num,8192])
#ground_truth = np.zeros([sample_num,2])
#for i in range(sample_num):
#    # sample from unique label
#    sample_label = shuffle_label[i%np.unique(Y).size]
#    # sample neg or pos
#    neg_or_pos = np.random.random_integers(0,1,1)
#    if neg_or_pos ==1:
#        temp_array_1= np.where(Y==sample_label)[0]
#        np.random.shuffle(temp_array_1)
#        indice1 = temp_array_1[0]
#        indice2 = temp_array_1[1]
#    else :
#        temp_array_1= np.where(Y==sample_label)[0]
#        temp_array_2= np.where(Y!=sample_label)[0]
#        np.random.shuffle(temp_array_1)
#        np.random.shuffle(temp_array_2)
#        indice1 = temp_array_1[0]
#        indice2 = temp_array_2[0]
#    X_1 = X[:,indice1]
#    X_2 = X[:,indice2]
#    ground_truth[i,neg_or_pos] = 1  
#    data[i,:] = np.concatenate((X_1,X_2),axis=0)
#

""" only deploy piece that's needed """
sample_num = 40000
count = 0
caffe.set_mode_gpu()
net = caffe.Net('auto_test.prototxt','./model_hard/mlp_iter_70000.000000',caffe.TEST)
for i in range(sample_num):
    # input is 4096 *2 vector concatenated to 8192*1 vector,  each vector individually has norm 1
    # truth is 2*1 
    out = net.forward()
    if net.blobs['label'].data[...].argmax() == out['prob'].argmax():
	count = count+1
    else:
        print out['prob'][0,1]
print count
