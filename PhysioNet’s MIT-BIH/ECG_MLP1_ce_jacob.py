import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from RobustDNN_loss import dZdX_loss2zs, dZdX_loss2zs_ref, dZdX_loss3zs, dZdX_loss3zs_ref, multi_margin_loss, dZdX_jacob
from RobustDNN_module import Linear
from Evaluate import test_adv
from ECG_Dataset import get_dataloader
from ECG_MLP1 import Net, main
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(0)
#%%
def train1(model, device, optimizer, dataloader, epoch, train_arg=None):
    model.train()#set model to training mode
    beta = train_arg["beta"]
    loss1_train=0
    loss2_train=0
    acc_train =0
    sample_count=0
    if epoch > 0:
        model.initialize_dead_kernel()
    model.zero_WoW()
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        model.zero_grad()
        model.normalize_kernel()
        #---------------------------
        Z = model(X)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y=Yp==Y
        #---------------------------
        loss1 = nnF.cross_entropy(Z, Y)
        loss1.backward(retain_graph=True)

        loss2 = dZdX_jacob(model, X, Y, norm=1)
        (beta*loss2).backward(retain_graph=True)
        L1_weight_decay(optimizer, 0.01)
        L2_weight_decay(optimizer, 0.01)
        optimizer.step()
        model.update_WoW()
        """
        loss2 = multi_margin_loss(Z[Yp_e_Y], Y[Yp_e_Y], margin=1, within_margin=True, reduction='counter')        
        loss3 = dZdX_loss3zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)
        loss4 = dZdX_loss3zs_ref(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)        
        loss5 = dZdX_loss2zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True, alpha=0)
        loss6 = dZdX_loss2zs_ref(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True, alpha=0)
        """
        loss1_train+=loss1.item()
        loss2_train+=loss2.item()
        """
        loss3_train+=loss3.item()
        loss4_train+=loss4.item()
        loss5_train+=loss5.item()
        loss6_train+=loss6.item()
        """
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss1: {:.6f}\tLoss2: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), 
                   loss1.item(),loss2.item()))
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    """
    loss3_train/=len(dataloader)
    loss4_train/=len(dataloader)
    loss5_train/=len(dataloader)
    loss6_train/=len(dataloader)
    """
    acc_train/=sample_count
    return (loss1_train,loss2_train), acc_train
#%%
def train2(model, device, optimizer, dataloader, epoch, train_arg=None):
    model.train()#set model to training mode
    beta = train_arg["beta"]
    loss1_train=0
    loss2_train=0
    acc_train =0
    sample_count=0
    if epoch > 0:
        model.initialize_dead_kernel()
    model.zero_WoW()
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        model.zero_grad()
        model.normalize_kernel()
        #---------------------------
        Z = model(X)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y=Yp==Y
        #---------------------------
        loss1 = nnF.cross_entropy(Z, Y)
        loss1.backward(retain_graph=True)

        loss2 = dZdX_jacob(model, X, Y, norm=2)
        (beta*loss2).backward(retain_graph=True)
        L1_weight_decay(optimizer, 0.01)
        L2_weight_decay(optimizer, 0.01)
        optimizer.step()
        model.update_WoW()
        """
        loss2 = multi_margin_loss(Z[Yp_e_Y], Y[Yp_e_Y], margin=1, within_margin=True, reduction='counter')        
        loss3 = dZdX_loss3zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)
        loss4 = dZdX_loss3zs_ref(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)        
        loss5 = dZdX_loss2zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True, alpha=0)
        loss6 = dZdX_loss2zs_ref(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True, alpha=0)
        """
        loss1_train+=loss1.item()
        loss2_train+=loss2.item()
        """
        loss3_train+=loss3.item()
        loss4_train+=loss4.item()
        loss5_train+=loss5.item()
        loss6_train+=loss6.item()
        """
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss1: {:.6f}\tLoss2: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), 
                   loss1.item(),loss2.item()))
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    """
    loss3_train/=len(dataloader)
    loss4_train/=len(dataloader)
    loss5_train/=len(dataloader)
    loss6_train/=len(dataloader)
    """
    acc_train/=sample_count
    return (loss1_train,loss2_train), acc_train

def L1_weight_decay(optimizer, rate):
    if rate <= 0:
        return
    with torch.no_grad():
        for g in optimizer.param_groups:
            lr=g['lr']
            for p in g['params']:
                if p.requires_grad == True:
                    p -= lr*rate*p.sign()
#%%
def L2_weight_decay(optimizer, rate):
    if rate <= 0:
        return
    with torch.no_grad():
        for g in optimizer.param_groups:
            lr=g['lr']
            for p in g['params']:
                if p.requires_grad == True:
                    p -= lr*rate*p
#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])
#%%
if __name__ == '__main__':
#%%
    for b in [0.9]:
        train_arg={}
        train_arg["jacob1"]=True
        train_arg['lr']=0.001
        train_arg["beta"] = b;
        train_arg['device'] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        main(batch_size=128, num=128, bias=True, loss_name='ce_beta'+str(b),
             epoch_start=50, epoch_end=50, train=train1, train_arg=train_arg, evaluate_model=True)
