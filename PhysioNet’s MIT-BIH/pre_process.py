# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:35:23 2019

@author: linhai
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_train=pd.read_csv('ecg/mitbih_train.csv', header=None)
df_test=pd.read_csv('ecg/mitbih_test.csv', header=None)
X_train=df_train.values[:,0:-1]
Y_train=df_train.values[:,-1].astype('int64')
X_test=df_test.values[:,0:-1]
Y_test=df_test.values[:,-1].astype('int64')

"""
t=np.linspace(0, 1.496, 187)
for c in range(0, 5):
    fig, ax = plt.subplots()
    for n in range(0, 5):
        idx=np.random.randint(0,10)
        ax.plot(t, X_train[Y_train==c][idx])
    ax.set_title('class'+str(c))
plt.show()
"""

 #%% split data into training set, validation set, testing set
import torch
rng=np.random.RandomState(0)
idxlist=np.arange(0, X_train.shape[0])
rng.shuffle(idxlist)
length =X_train.shape[0]
print ("x train shape is ", X_train.shape[0], " ", X_train.shape[1])
print ("train set length is ",int(0.8*length) )
print ("val set length is ",int(0.2*length) )
print ("test size is ", X_test.shape[0])
idxlist_train=idxlist[0:int(0.8*length)]   # 80% for training
idxlist_val=idxlist[int(0.8*length):] # 20% for validation
#  70043 training, 17510 val, 21892 test
data={}
data['X_train']=torch.tensor(X_train[idxlist_train,:], dtype=torch.float32)
data['Y_train']=torch.tensor(Y_train[idxlist_train], dtype=torch.int64)
data['X_val']=torch.tensor(X_train[idxlist_val,:], dtype=torch.float32)
data['Y_val']=torch.tensor(Y_train[idxlist_val], dtype=torch.int64)
data['X_test']=torch.tensor(X_test, dtype=torch.float32)
data['Y_test']=torch.tensor(Y_test, dtype=torch.int64)
torch.save(data, 'mitbih_data.pt')
