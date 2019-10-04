import csv
import math
import copy
import time
import numpy as np
import pandas as ps
from collections import Counter
from numpy import *
feature = []

#Classifier decision tree using ID3 algorithm

#normalize the entire dataset prior to learning using min-max normalization 
def normalize(matrix):
    a= np.array(matrix)
    a = a.astype(np.float)
    #print(a)
    #print("Before normalizing")
    b = np.apply_along_axis(lambda x: (x-np.min(x))/float(np.max(x)-np.min(x)),0,a)
    return b
    #print(b)
    
# reading from the file using numpy genfromtxt
def load_csv(file):
    X = genfromtxt(file, delimiter=",",dtype=str)
    return (X)

#method to randomly shuffle the array
def random_numpy_array(ar):
    np.random.shuffle(ar)
    arr = ar
    #print(arr)
    return arr

#Normalize the data and generate the training labels,training features, test labels and test training
def generate_set(X):
    #print(X.shape[0])
    Y = X[:,-1]
    j = Y.reshape(len(Y),1)
    #print("J is",j)
    new_X = X[:,:-1]
    #normalize the data step
    normalized_X = normalize(new_X)
    normalized_final_X = np.concatenate((normalized_X,j),axis=1)
    X = normalized_final_X
    size_of_rows = X.shape[0]
    num_test = round(0.1*(X.shape[0]))
    start = 0
    end = num_test
    test_attri_list =[]
    test_class_names_list =[]
    training_attri_list = []
    training_class_names_list = []
    for i in range(10):
        X_test = X[start:end , :]
        tmp1 = X[:start, :]
        tmp2 = X[end:, :]
        X_training = np.concatenate((tmp1, tmp2), axis=0)
        #X_training = X[:start,:]+ X[end: , :]
    #print("Before normalizing",X_test)
        y_test = X_test[:, -1]
        y_test = y_test.flatten()
        y_training = X_training[:,-1]
        y_training = y_training.flatten()
        #y_train = y_training.astype(np.float)
        #y_test = y_test.astype(np.float)
        X_test = X_test[:,:-1]
        X_training = X_training[:,:-1]
        X_test = X_test.astype(np.float)
        X_training = X_training.astype(np.float)
        test_attri_list.append(X_test)
        test_class_names_list.append(y_test)
        training_attri_list.append(X_training)
        training_class_names_list.append(y_training)
        #print("start is",start)
        #print("end is",end)
        start = end
        end = end+num_test
    return test_attri_list,test_class_names_list,training_attri_list,training_class_names_list#(X_test,y_test,X_training,y_train)

# Iterative Dichotomiser 3 entropy calculation
def entropy(y):
    class_freq = {}
    attribute_entropy = 0.0
    for i in y:
        if class_freq.has_key(i):
            class_freq[i] += 1
        else:
            class_freq[i] = 1
    #print(class_freq)
    for freq in class_freq.values():
        attribute_entropy += (-freq/float(len(y))) * math.log(freq/float(len(y)),2)
    #print(attribute_entropy)
    return attribute_entropy

#calculating the predicited accuracy
def accuracy_for_predicted_values(test_class_names1,l):
    true_count = 0
    false_count = 0
    for i in range(len(test_class_names1)):
        if(test_class_names1[i] == l[i]):
            true_count += 1
        else:
            false_count += 1
    return true_count, false_count, float(true_count) / len(l)

#build a dictionary where the key is the class label and values are the features which belong to that class.
def build_dict_of_attributes_with_class_values(X,y):
    dict_of_attri_class_values = {}
    fea_list =[]
    for i in xrange(X.shape[1]):
        fea = i
        l = X[:,i]
        #print(l)
        attribute_list =[]
        count = 0
        for j in l:
            attribute_value = []
            attribute_value.append(j)
            attribute_value.append(y[count])
            attribute_list.append(attribute_value)
            count += 1
        dict_of_attri_class_values[fea]= attribute_list
        fea_list.append(fea)
    return dict_of_attri_class_values,fea_list

def return_features(Y):
    return feature
#Class node and explanation is self explaination
class Node(object):
    def __init__(self, val, lchild, rchild,the,leaf):
    #def __init__(self,val,):
        self.root_value = val
        self.root_left = lchild
        self.root_right = rchild
        self.theta = the
        self.leaf = leaf

    #method to identify if the node is leaf
    def is_leaf(self):
        return self.leaf

    #method to return threshold value
    def ret_thetha(self):
        return self.theta

    def ret_root_value(self):
        return self.root_value

    def ret_llist(self):
        return self.root_left

    def ret_rlist(self):
        return self.root_right

    def __repr__(self):
        return "(%r, %r, %r, %r)" %(self.root_value,self.root_left,self.root_right,self.theta)
