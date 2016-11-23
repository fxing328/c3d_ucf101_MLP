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


if __name__ == '__main__':
  #   file1 = pkl.load(open('feature_c3d_train_fc6','rb'))
  # # # construction training set for triplet loss
  #   X = np.matrix(file1['data'])
  #   Y = np.array(file1['label'])

  #   X = preprocessing.normalize(X,norm='l2').T

  #   file2 = pkl.load(open('feature_c3d_test_fc6','rb'))
  #   #    # construction training set for triplet loss
  #   X_test = np.matrix(file2['data'])
  #   Y_test = np.array(file2['label'])

  #   X_test = preprocessing.normalize(X_test,norm='l2').T
 
  #   sample_num = 100000 
  #   f= h5py.File('train.h5','w')
  #   f.create_dataset('data',(sample_num,8192),dtype='f8')
  #   f.create_dataset('label',(sample_num,2),dtype='f4')

 
  #   X_1=  np.zeros([X.shape[0]])
  #   X_2=  np.zeros([X.shape[0]])
  #   train_label = np.zeros(2)
  #   shuffle_label = np.unique(Y)
  #   np.random.shuffle(shuffle_label)#
  #   for i in range(sample_num):
  #      # sample from unique label
  #      sample_label = shuffle_label[i%np.unique(Y).size]
  #      # sample neg or pos
  #      neg_or_pos = np.random.random_integers(0,1,1)
  #      if neg_or_pos ==1:
  #          temp_array_1= np.where(Y==sample_label)[0]
  #          np.random.shuffle(temp_array_1)
  #          indice1 = temp_array_1[0]
  #          indice2 = temp_array_1[1]
  #      else :
  #          temp_array_1= np.where(Y==sample_label)[0]
  #          temp_array_2= np.where(Y!=sample_label)[0]
  #          np.random.shuffle(temp_array_1)
  #          np.random.shuffle(temp_array_2)
  #          indice1 = temp_array_1[0]
  #          indice2 = temp_array_2[0]
  #      print i
  #      X_1 = X[:,indice1]
  #      X_2 = X[:,indice2]
  #      train_label = np.zeros(2)
  #      train_label[neg_or_pos]=1
  #      f['data'][i] = np.concatenate((X_1,X_2),axis=0)
  #      f['label'][i] = train_label 
  #   f.close()

 
  #   sample_num = 10000
  #   f1= h5py.File('test.h5','w')
  #   f1.create_dataset('data',(sample_num,8192),dtype='f8')
  #   f1.create_dataset('label',(sample_num,2),dtype='f4')

  #   X_1=  np.zeros([X.shape[0]])
  #   X_2=  np.zeros([X.shape[0]])
  #   test_label = np.zeros(sample_num)
  #   shuffle_label = np.unique(Y_test)
  #   np.random.shuffle(shuffle_label)
  #   for i in range(sample_num):
  #      # sample from unique label
  #      sample_label = shuffle_label[i%np.unique(Y).size]
  #      # sample neg or pos
  #      neg_or_pos = np.random.random_integers(0,1,1)
  #      if neg_or_pos ==1:
  #          temp_array_1= np.where(Y_test==sample_label)[0]
  #          np.random.shuffle(temp_array_1)
  #          indice1 = temp_array_1[0]
  #          indice2 = temp_array_1[1]
  #      else :
  #          temp_array_1= np.where(Y_test==sample_label)[0]
  #          temp_array_2= np.where(Y_test!=sample_label)[0]
  #          np.random.shuffle(temp_array_1)
  #          np.random.shuffle(temp_array_2)
  #          indice1 = temp_array_1[0]
  #          indice2 = temp_array_2[0]
  #      print i
  #      X_1 = X[:,indice1]
  #      X_2 = X[:,indice2]
  #      test_label = np.zeros(2)
  #      test_label[neg_or_pos]=1
  #      f1['data'][i] = np.concatenate((X_1,X_2),axis=0)
  #      f1['label'][i] = test_label

  #      #
  #   f1.close()

    sample_num = 100000
  #   with open('./auto_train.prototxt', 'w') as f:
  #       f.write(str(net('train.h5list', 50)))
  #   with open('auto_test.prototxt', 'w') as f:
  #       f.write(str(net('test.h5list', 20)))

    caffe.set_device(0)
    caffe.set_mode_gpu()
    
    net = caffe.Net('./auto_train1.prototxt','./model/mlp_iter_99000.000000',caffe.TEST)

    
    
    #solver.step(1)

    niter = 10
    test_interval = 1000
    #train_loss = zeros(niter)
    test_acc = zeros(int(np.ceil(niter * 1.0 / test_interval)))
    print len(test_acc)
    output = zeros((niter, 2, 2))

    # The main solver loop
    for it in range(niter):
        # solver.step(1)  # SGD by Caffe
        # train_loss[it] = solver.net.blobs['loss'].data
    
        if it % test_interval == 0:
            print 'Iteration', it, 'testing...'
	    correct = 0
            data = net.blobs['ip3'].data
            label = net.blobs['label'].data
	    pdb.set_trace()
            for test_it in range(sample_num):
                out = net.forward()
                # Positive values map to label 1, while negative values map to label 0
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        if data[i][j] > 0 and label[i][j] == 1:
                            correct += 1
                        elif data[i][j] <= 0 and label[i][j] == 0:
                            correct += 1
            test_acc[int(it / test_interval)] = correct * 1.0 / (len(data) * len(data[0]) * sample_num)
    	   #solver.net.save('./model/mlp_iter_%f'%(it))
    scipy_io.savemat('curve.mat',{'test':test_acc}) 
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
