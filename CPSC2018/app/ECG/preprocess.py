import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
#from CPSC2018_Dataset import pad, pad_sequence
#%%
path='../../data/CPSC2018/train/'

df=pd.read_csv(path+'/REFERENCE.csv')
#%% see if the data can fit in memory, it seems ok
signal=[]
for n in range(df.shape[0]):
    name=path+df['Recording'][n]+'.mat'
    temp=loadmat(name)
    ecg=temp['ECG'][0][0][2]
    ecg=ecg.astype('float32')
    signal.append(ecg)
#%% check if there is any bad ecg signal
length=[]
for n in range(df.shape[0]):
    length.append(signal[n].shape[1])
length=np.array(length)
#3000 to 72000
#%np.sum(length>30720) 26
#%%
plt.hist(length, bins=100)
#%% only do this for mini-batch
#signal, length=pad_sequence(signal)
#%% class banlance
plt.hist(df['First_label'].values, bins=9)
#%% mark the sample which has more than one label
#477 samples
flaglist=(df['Second_label'].isnull().values==False).astype('int64')
#%% label
labellist=df['First_label'].values-1
#%% train, val, test, split, total=6877
#select 5 samples per class for validation
#select 50 samples per class for test
rng=np.random.RandomState(0)
idxlist=np.arange(0, 6877, 1)
#remove the 477 samples
#idxlist2=idxlist[flaglist==1]
idxlist=idxlist[flaglist==0]
#%%
rng.shuffle(idxlist)
idxlist_val=[]
counter_val=np.zeros(9)
for n in range(len(idxlist)):
    id=idxlist[n]
    label=labellist[id]
    if counter_val[label] < 5:
        idxlist_val.append(id)
        counter_val[label]+=1
idxlist=list(set(idxlist)-set(idxlist_val))
#%%
for n in range(9):
    rng.shuffle(idxlist)
idxlist_test=[]
counter_test=np.zeros(9)
for n in range(len(idxlist)):
    id=idxlist[n]
    label=labellist[id]
    if counter_test[label] < 50:
        idxlist_test.append(id)
        counter_test[label]+=1
idxlist_train=list(set(idxlist)-set(idxlist_test))
#idxlist_train=list(idxlist_train)+list(idxlist2)
#%%
fig, ax=plt.subplots(3,1, sharex=True)
ax[0].hist(length[idxlist_train], bins=100)
ax[1].hist(length[idxlist_val], bins=100)
ax[2].hist(length[idxlist_test], bins=100)
#%% create df_train, df_val, df_test
df_train=df.iloc[idxlist_train]
df_val=df.iloc[idxlist_val]
df_test=df.iloc[idxlist_test]
df_train.to_csv(path+'/train.csv', index=False)
df_val.to_csv(path+'/val.csv', index=False)
df_test.to_csv(path+'/test.csv', index=False)
#%% verify


label_val=df['First_label'].values[idxlist_val]
plt.hist(label_val, bins=9)
#%% now check the 'validation_set'
pat_val='C:/Research/ICMLA20/data/CPSC2018/train/'
dfv=pd.read_csv(pat_val+'REFERENCE.csv')
#%%
signal_v=[]
for n in range(dfv.shape[0]):
    name=pat_val+dfv['Recording'][n]+'.mat'
    temp=loadmat(name)
    ecg=temp['ECG'][0][0][2]
    ecg=ecg.astype('float32')
    signal_v.append(ecg)
#%% compare signal and signal_v
#validation_set is from train set..
error=[]
for n in range(len(signal_v)):
    if n != 201:
        error.append(np.mean(np.abs(signal[n]-signal_v[n])))
error=np.array(error)
#%% check if there is any bad ecg signal
length_v=[]
for n in range(dfv.shape[0]):
    length_v.append(signal_v[n].shape[1])
length_v=np.array(length_v)
#4999 to 32000
