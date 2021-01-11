import numpy as np
import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
from ClassBalancedSampler import ClassBalancedSampler
#%%
class MyDataset(torch_dataset):
    def __init__(self, X, Y, Mask=None, return_idx=False):
        self.X=X.detach()
        self.Y=Y.detach()
        self.return_idx=return_idx
        self.Mask=None
        if Mask is not None:
            self.Mask=Mask.detach()
    def __len__(self):
        #return the number of data points
        return self.X.shape[0]
    def __getitem__(self, idx):
        if self.Mask is None:
            if self.return_idx == False:
                return self.X[idx,:], self.Y[idx]
            else:
                return self.X[idx,:], self.Y[idx], idx
        else:
            if self.return_idx == False:
                return self.X[idx], self.Y[idx], self.Mask[idx]
            else:
                return self.X[idx], self.Y[idx], self.Mask[idx], idx
#%%
def get_dataloader(batch_size=128, num_workers=0):
    data= torch.load('mitbih_data.pt')
    dataset_train = MyDataset(data['X_train'], data['Y_train'])
    dataset_val = MyDataset(data['X_val'], data['Y_val'])
    dataset_test = MyDataset(data['X_test'], data['Y_test'])
    print ("train data :", data['Y_train'].shape)
    print ("val data :", data['Y_val'].shape)
    print ("test data :", data['Y_test'].shape)
    sample_count_train=np.zeros(5)
    for n in range(0, 5):
        sample_count_train[n]=torch.sum(data['Y_train']==n).item()
    sample_count_val=np.zeros(5)
    for n in range(0, 5):
        sample_count_val[n]=torch.sum(data['Y_val']==n).item()
    sample_count_test=np.zeros(5)
    for n in range(0, 5):
        sample_count_test[n]=torch.sum(data['Y_test']==n).item()
    print('sample_count_train', sample_count_train)
    print('sample_count_val', sample_count_val)
    print('sample_count_test', sample_count_test)

    sampler_train=ClassBalancedSampler(data['Y_train'].numpy(), True)
    sampler_val=ClassBalancedSampler(data['Y_val'].numpy(), False)
    sampler_test=ClassBalancedSampler(data['Y_test'].numpy(), False)
    loader_train = torch_dataloader(dataset_train, batch_size=batch_size,  sampler=sampler_train, num_workers=0)
    loader_val = torch_dataloader(dataset_val, batch_size=batch_size, sampler=sampler_val, num_workers=0)
    loader_test = torch_dataloader(dataset_test, batch_size=batch_size, sampler=sampler_test, num_workers=0)

    return loader_train, loader_val, loader_test
#%%
def get_dataloader_bba(batch_size=128, num_workers=0):
    data= torch.load('mitbih_data.pt')
    sample_count_train=np.zeros(5)
    for n in range(0, 5):
        sample_count_train[n]=torch.sum(data['Y_train']==n).item()
    sample_count_val=np.zeros(5)
    for n in range(0, 5):
        sample_count_val[n]=torch.sum(data['Y_val']==n).item()
    sample_count_test=np.zeros(5)
    for n in range(0, 5):
        sample_count_test[n]=torch.sum(data['Y_test']==n).item()
    print('sample_count_train', sample_count_train)
    print('sample_count_val', sample_count_val)
    print('sample_count_test', sample_count_test)

    print('take the first 162 samples from each class (testing set) for bba test')

    X_test=[]
    Y_test=[]
    for n in range(0, 5):
        X_n=data['X_test'][data['Y_test']==n][0:162]
        X_test.append(X_n)
        Y_test.append(n*torch.ones(X_n.size(0), dtype=torch.int64))
    X_test=torch.cat(X_test, dim=0)
    Y_test=torch.cat(Y_test, dim=0)
    print(X_test.shape)
    print(Y_test.shape)
    dataset_test = MyDataset(X_test, Y_test)    
    loader_bba = torch_dataloader(dataset_test, batch_size=batch_size, num_workers=0)
    return loader_bba
#%%