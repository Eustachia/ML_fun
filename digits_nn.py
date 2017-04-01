# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:23:27 2017

@author: eust_abbondanza
"""
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_file='J:\\Eustachia\\KAGGLE\\digits\\train.csv'
test_file='J:\\Eustachia\\KAGGLE\\digits\\test.csv'
train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)
labels=train_data['label'].values
train=train_data.drop('label', axis=1).values  #train_data.ix[:, 1:]
X_std=StandardScaler().fit_transform(train)
Y_std=StandardScaler().fit_transform(test_data.values)
#print (labels[0:5])
#enc = OneHotEncoder(n_values=len(np.unique(labels)))
new_labels=np.zeros([len(labels), len(np.unique(labels))])
for i in range(0, len(labels)):
    new_labels[i, labels[i]]=1#enc.transform(np.asarray(labels)).toarray()
#print (new_labels[0:5])

# Extract data from dataframe
#data = X_std.as_matrix()
# Split data into training set and validation set
#y = data[:, 0]
X = X_std[:, 1:].astype(np.float64)
Y = Y_std[:, 1:].astype(np.float64)
train_num = 41000
val_num = 1000
X_train, y_train = X[:train_num], new_labels[:train_num]
X_val, y_val = X[train_num:], new_labels[train_num:]

print(X_train.shape, y_train.shape, X_train.dtype, y_train.dtype)
print(X_val.shape, y_val.shape, X_val.dtype, y_val.dtype)
N_f=X_train.shape[1]
N_s=X_train.shape[0]

layer1=500
layer2=10

def initialize_layers(layer1, layer2, N_samples, N_features):
  #  global W1, W2, b1, b2
    
    W1=0.001*np.random.rand(N_f, layer1)
    b1=np.zeros(layer1)
    W2=0.001*np.random.rand(layer1, layer2)
    b2=np.zeros(layer2)
    print (W1.shape, W2.shape, b1.shape, b2.shape)
    return W1, W2, b1, b2
    


def softmax(scores):
    #exponentiate scores
    exp_sc=np.exp(scores)
    
    return exp_sc / exp_sc.sum(axis=1, keepdims=True)
    
def sigmoid(input_data):
    return 1 / (1+np.exp(np.multiply(input_data, -1))) #try also tanh !
    
    
def forward_pass(data, W1, W2, b1, b2):
  #  print (data.dot(W1).shape)
   # print (b1.shape)
    #print ((data.dot(W1)+b1).shape)
    output1=sigmoid(data.dot(W1)+b1)
    
    scores=softmax(output1.dot(W2)+b2)
    
    
    return output1, scores
    
def classif_acc(labels, pred_labels):
    acc=0
    for i in range(0, len(pred_labels)):
        if pred_labels[i]==labels[i]:
            acc+=1
    return acc / len(pred_labels)
    
def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]
    ret4 = Z.T.dot(T - Y)
    # assert(np.abs(ret1 - ret4).sum() < 0.00001)

    return ret4
    
def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret2 = X.T.dot(dZ)

    # assert(np.abs(ret1 - ret2).sum() < 0.00001)

    return ret2
    
def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)


def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def cost(T, Y):
    tot = T * np.log(Y)
    return -np.mean(tot) #tot.sum()

    
    
def main():
    
    W1, W2, b1, b2 = initialize_layers(layer1, layer2, N_s, N_f) 
    
    learning_rate = 10e-6
    costs = []
    costs_test=[]
    
    for epoch in range(0, 1000):
        hidden, output = forward_pass(X_train, W1, W2,b1, b2)
     #   print (hidden.shape, output.shape)
        hidden_test, output_test = forward_pass(X_val, W1, W2,b1, b2)
     #   print (hidden_test.shape, output_test.shape)
        if epoch % 100 == 0:
            c = cost(y_train, output)
            c_test = cost(y_val, output_test)
         #   print (output[0:5])
            P = np.argmax(output, axis=1)
            r = classif_acc(P, labels[:train_num])
            
            P_test = np.argmax(output_test, axis=1)
            r_test = classif_acc(P_test, labels[train_num:])
            print ("cost:", c, "classification_rate:", r)
            print ("cost test:", c_test, "classification_rate_test:", r_test)
            costs.append(c)
            costs_test.append(c_test)
        # this is gradient ASCENT, not DESCENT
        # be comfortable with both!
        # oldW2 = W2.copy()
        W2 += learning_rate * derivative_w2(hidden, y_train, output)
        b2 += learning_rate * derivative_b2(y_train, output)
        W1 += learning_rate * derivative_w1(X_train, hidden, y_train, output, W2)
        b1 += learning_rate * derivative_b1(y_train, output, W2, hidden)
    plt.figure(1)
    plt.plot(costs)
    plt.show()
    plt.figure(2)
    plt.plot(costs_test)
    plt.show()
    
    hidden_test1, output_test1 = forward_pass(Y, W1, W2,b1, b2)    
    
    my_prediction=np.argmax(output_test1, axis=1)
    
    my_solution = pd.DataFrame(my_prediction, columns = ["Label"])
    my_solution.index += 1 
    my_solution.head()
    my_solution.to_csv("my_solution_nn1.csv", index_label = ["ImageId"])

    
if __name__ == '__main__':
    main()
