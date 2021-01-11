import sys
sys.path.append('../../core')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
from ClassBalancedSampler import ClassBalancedSampler
from sklearn.preprocessing import MaxAbsScaler
#%%
def pad(x, t_max, rand_pad):
    #x: 12 x T
    t=x.shape[1]
    if isinstance(x, torch.Tensor):
        xp=torch.zeros((x.shape[0], t_max), dtype=x.dtype, device=x.device)
        mask=torch.zeros((1, t_max), dtype=x.dtype, device=x.device)
    else:
        xp=np.zeros((x.shape[0], t_max), dtype=x.dtype)
        mask=np.zeros((1, t_max), dtype=x.dtype)
    if rand_pad == False:
        xp[:,t_max-t:]=x
        mask[:,t_max-t:]=1
    else:
        idx=t_max-t-int((np.random.rand())*3072)
        #idx=int(np.random.rand()*(t_max-t)+0.5)
        xp[:,idx:idx+t]=x
        mask[:,idx:idx+t]=1
    return xp, mask
#%%
def pad_sequence(x, t_max, rand_pad):
    # x=[x1, x2, ...,x_n,...]
    # x_n: 12 x t
    xp=[]
    mask=[]
    for x_n in x:
        out = pad(x_n, t_max, rand_pad)
        xp.append(out[0].view(1, *out[0].shape))
        mask.append(out[1].view(1, *out[1].shape))
    if isinstance(x, torch.Tensor):
        xp=torch.cat(xp, dim=0)
        mask=torch.cat(mask, dim=0)
    else:
        xp=np.concatenate(xp, axis=0)
        mask=np.concatenate(mask, axis=0)
    return xp, mask
#%%
'''
# 3000 has 5 seconds while the longest of 72000
# pad 3000  to 3072=1024*3
# pad 72000 to 73728=3072*24
# pad any sequence to 3072*n
if 0:
    model=Resnet18()

    x=torch.rand(1,12,3072*24)
    z=model(x)

    x=torch.rand(12, 8679)
    x,m=pad(x)
    z=model(x)

    x1=torch.rand(12, 8900)
    x2=torch.rand(12, 9865)
    x,m=pad_sequence([x1, x2], 73728)
    z=model(x)
'''
#%%
class MyDataset(torch_dataset):
    def __init__(self, signal, label, rand_pad):
        self.signal=signal
        self.label=label
        self.rand_pad=rand_pad
    def __len__(self):
        return len(self.signal)
    def __getitem__(self, idx):
        x = self.signal[idx]
        t_max=30720+3072
        if x.shape[1]>30720:
            x=x[:,0:30720]
        #xmax=np.max(np.abs(x))
        #x=x/xmax
        scaler = MaxAbsScaler()
        #temp_x = x.view(x.size(1),x.size(2)).numpy()
        # remove lead3,4,5,6
        m = [True, True, False, False, False, False, True,True,True,True,True,True]
        # do absolute value scaling
        x = x[m]
        x = scaler.fit_transform(x.T).T
        x=torch.tensor(x, dtype=torch.float32)
        x, mask = pad(x, t_max=t_max, rand_pad=self.rand_pad)
        y = torch.tensor(self.label[idx], dtype=torch.int64)
        return x, y, mask, idx
#%%
def load_signal(df, path):
    signal=[]
    for n in range(df.shape[0]):
        name=path+df['Recording'][n]+'.mat'
        temp=loadmat(name)
        ecg=temp['ECG'][0][0][2]
        ecg=ecg.astype('float32')
        signal.append(ecg)
    return signal
#%%
def get_dataloader(batch_size=64, num_workers=0, rand_pad=False, path='../../data/CPSC2018/'):
    df_train=pd.read_csv(path+'train/train.csv')
    df_val=pd.read_csv(path+'train/val.csv')
    df_test=pd.read_csv(path+'train/test.csv')
    #df_train = pd.concat([df_train_0, df_test_0], ignore_index=True)
    print('load signal:', path)
    signal_train=load_signal(df_train, path+'train/')
    signal_val=load_signal(df_val, path+'train/')
    signal_test=load_signal(df_test, path+'train/')
    print('load signal: completed')
    #label in csv is from 1 to 9
    label_train=df_train['First_label'].values-1
    label_val=df_val['First_label'].values-1
    label_test=df_test['First_label'].values-1

    dataset_train = MyDataset(signal_train, label_train, rand_pad=rand_pad)
    dataset_val = MyDataset(signal_val, label_val, rand_pad=False)
    dataset_test = MyDataset(signal_test, label_test, rand_pad=False)

    sampler_train = ClassBalancedSampler(label_train, True)
    loader_train = torch_dataloader(dataset_train, batch_size=batch_size,
                                    num_workers=num_workers, sampler=sampler_train, pin_memory=True)
    loader_val = torch_dataloader(dataset_val, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=False, pin_memory=True)
    
    loader_test = torch_dataloader(dataset_test, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=False, pin_memory=True)

    return loader_train, loader_val, loader_test

#%%


