import matplotlib.pyplot as plt
import numpy as np
import pdb
import cPickle as pkl
from sklearn import preprocessing
import h5py 
import copy
import math
import matplotlib
import caffe, h5py
import scipy.io as scipy_io
from pylab import *
from caffe import layers as L
from sklearn.svm import SVC
from sklearn import grid_search

caffe.set_mode_gpu()
net_old = caffe.Net('./auto_test.prototxt','./model_n/mlp_iter_99000.000000',caffe.TEST)
def get_prob(net_old,data1,data2):
    data = np.concatenate([data1,data2],axis=0)
    net_old.blobs['data'].data[...] = data
    out = net_old.forward()
    return out['prob'][0,1] # return confidence level


def net(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.ip1 = L.InnerProduct(n.data, num_output=1024, weight_filler=dict(type='xavier'))
    n.relu1 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu1, num_output=1024, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.ip2, in_place=True)
    n.ip3 = L.InnerProduct(n.relu2, num_output=2, weight_filler=dict(type='xavier'))
    n.loss = L.SigmoidCrossEntropyLoss(n.ip3, n.label)
    return n.to_proto()

def train_svm(data,label):
    clf = SVC(kernel = 'linear',probability= True)
    Cgrid = np.linspace(0.1,10,10).tolist()
    parameters = {'kernel':['linear'], 'C':Cgrid}
    cv_clf = grid_search.GridSearchCV(clf,parameters)
    cv_clf.fit(data,label)
    clf = cv_clf
    return clf 



if __name__ == '__main__':

#    f_train = h5py.File('train_hard.h5','r')
#    train_data = np.asarray(f_train['data'])
#    train_label = np.asarray(f_train['label']).argmax(axis=1)
#    train_label = train_label[0:train_data.shape[0]:5]
#    train_data = train_data[0:train_data.shape[0]:5,:]
#    f_test = h5py.File('test_hard.h5','r')
#    test_data = np.asarray(f_test['data'])
#    test_label = np.asarray(f_test['label']).argmax(axis=1)
#    clf = train_svm(train_data,train_label) 
#    print clf.score(test_data,test_label)
#    pkl.dump(clf,open('svm_classifier','w'))
#
    file1 = pkl.load(open('./feature_c3d_train_fc6','rb'))
   # # construction training set for triplet loss
    X = np.matrix(file1['data']).T
    Y = np.array(file1['label'])

    X = preprocessing.normalize(X,norm='l2').T

 #   file2 = pkl.load(open('../feature_VGG_test_all_fc6','rb'))
 #   #    # construction training set for triplet loss
 #   X_test = np.matrix(file2['data'])
 #   Y_test = np.array(file2['label'])

 #   X_test = preprocessing.normalize(X_test,norm='l2').T
 #
    sample_num = 200000
    f= h5py.File('train_hard.h5','w')
    f.create_dataset('data',(sample_num,4096),dtype='f8')
    f.create_dataset('label',(sample_num,2),dtype='f4')
    f.create_dataset('sample_weight',(sample_num,2),dtype='f4')
 
    X_1=  np.zeros([X.shape[0]])
    X_2=  np.zeros([X.shape[0]])
    train_label = np.zeros(2)
    shuffle_label = np.unique(Y)
    np.random.shuffle(shuffle_label)#
    
    # hard negative mining
    hard_time =10 #select most similar hard negative same in #num trials
    for i in range(sample_num):
        # sample from unique label
        sample_label = shuffle_label[i%np.unique(Y).size]
        # sample neg or pos
        neg_or_pos = np.random.random_integers(0,1,1)
        if neg_or_pos ==1:
            temp_array_1= np.where(Y==sample_label)[0]
            np.random.shuffle(temp_array_1)
            indice1 = temp_array_1[0]
            indice2 = temp_array_1[1]
        else :
            temp_array_1= np.where(Y==sample_label)[0]
            np.random.shuffle(temp_array_1)
            temp_array_2= np.where(Y!=sample_label)[0]
            np.random.shuffle(temp_array_2)
            
            max_prob = 0
            for trail in range(hard_time):
                pos_id = temp_array_1[trail%temp_array_1.size]
                neg_id = temp_array_2[trail%temp_array_2.size]
                X_1 = X[:,pos_id]
                X_2 = X[:,neg_id]
                prob_test = get_prob(net_old,X_1,X_2)
                if max_prob < prob_test:
                    max_prob = prob_test
                    indice1 = pos_id
                    indice2 = neg_id
            print 'hard_negtive_prob: '+str(max_prob)

        print i
        X_1 = X[:,indice1]
        X_2 = X[:,indice2]
        train_label = np.zeros(2)
        train_label[neg_or_pos]=1
        f['sample_weight'][i] = np.array([0.7,0.3])
        f['data'][i] = np.concatenate((X_1,X_2),axis=0)
        f['label'][i] = train_label 
    f.close()

 
#    sample_num = 10000
#    f1= h5py.File('test_hard.h5','w')
#    f1.create_dataset('data',(sample_num,8192),dtype='f8')
#    f1.create_dataset('label',(sample_num,2),dtype='f4')
#    f1.create_dataset('sample_weight',(sample_num,2),dtype='f4')
#    
#    X_1=  np.zeros([X.shape[0]])
#    X_2=  np.zeros([X.shape[0]])
#    test_label = np.zeros(sample_num)
#    shuffle_label = np.unique(Y_test)
#    np.random.shuffle(shuffle_label)
#    for i in range(sample_num):
#        # sample from unique label
#        sample_label = shuffle_label[i%np.unique(Y_test).size]
#        # sample neg or pos
#        neg_or_pos = np.random.random_integers(0,1,1)
#        if neg_or_pos ==1:
#            temp_array_1= np.where(Y_test==sample_label)[0]
#            np.random.shuffle(temp_array_1)
#            indice1 = temp_array_1[0]
#            indice2 = temp_array_1[1]
#        else :
#            temp_array_1= np.where(Y_test==sample_label)[0]
#            temp_array_2= np.where(Y_test!=sample_label)[0]
#            np.random.shuffle(temp_array_1)
#            np.random.shuffle(temp_array_2)
#            indice1 = temp_array_1[0]
#            indice2 = temp_array_2[0]
#        print i
#        X_1 = X[:,indice1]
#        X_2 = X[:,indice2]
#        test_label = np.zeros(2)
#        test_label[neg_or_pos]=1
#        f1['data'][i] = np.concatenate((X_1,X_2),axis=0)
#        f1['label'][i] = test_label
#        f1['sample_weight'][i] = np.array([0.7,0.3])
#
#        #
#    f1.close()
#

    sample_num = 10000
    #with open('./auto_train.prototxt', 'w') as f:
    #    f.write(str(net('train.h5list', 50)))
    #with open('auto_test.prototxt', 'w') as f:
    #    f.write(str(net('test.h5list', 20)))

    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('auto_solver.prototxt')
    solver.net.copy_from('./model_n/mlp_iter_99000.000000')
    solver.net.forward()
    solver.test_nets[0].forward()
    solver.step(1)

    niter = 1000000
    test_interval = 10000
    train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
    print len(test_acc)
    output = zeros((niter, 2, 2))

    # The main solver loop
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        train_loss[it] = solver.net.blobs['loss'].data
    
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
	    correct = 0
            data = solver.test_nets[0].blobs['ip3'].data
            label = solver.test_nets[0].blobs['label'].data
            for test_it in range(sample_num):
                solver.test_nets[0].forward()
                # Positive values map to label 1, while negative values map to label 0
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if data[i][j] > 0 and label[i][j] == 1:
                            correct += 1
                        elif data[i][j] <= 0 and label[i][j] == 0:
                            correct += 1
            test_acc[int(it / test_interval)] = correct * 1.0 / (len(data) * len(data[0]) * sample_num)
    	    solver.net.save('./model_hard/mlp_iter_%f'%(it))
    scipy_io.savemat('curve.mat',{'train':train_loss,'test':test_acc}) 
    #_,ax1 = plt.subplots(2,1,1)
    #ax2 = ax1.twinx()
    #ax1.plot(arange(niter), train_loss)
    #ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    #ax1.set_xlabel('iteration')
    #ax1.set_ylabel('train loss')
    #ax2.set_ylabel('test accuracy')
    #_.plt.savefig('converge.png')
    #
    ## Check the result of last batch
    #print solver.test_nets[0].blobs['ip3'].data
    #print solver.test_nets[0].blobs['label'].data
