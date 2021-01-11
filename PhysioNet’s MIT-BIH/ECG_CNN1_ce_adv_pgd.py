import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from RobustDNN_loss import dZdX_loss2zs, dZdX_loss2zs_ref, dZdX_loss3zs, dZdX_loss3zs_ref, multi_margin_loss
from Evaluate import test_adv, pgd_attack
from ECG_Dataset import get_dataloader
from ECG_CNN1 import Net, main
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
def save_checkpoint(filename, model, optimizer, result, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)
#%%
def train(model, device, optimizer, dataloader, epoch, train_arg):
    model.train()#set model to training mode
    noise=train_arg['noise']
    loss1_train=0
    loss2_train=0
    loss3_train=0
    loss4_train=0
    loss5_train=0
    loss6_train=0
    acc_train =0
    sample_count=0
    if epoch > 0:
        model.initialize_dead_kernel()
    model.zero_WoW()
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        Xn = pgd_attack(model, X, Y, noise, max_iter=10, step=0.01)
        #---------------------------
        model.zero_grad()
        model.normalize_kernel()
        Z = model(X)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y=Yp==Y
        Zn = model(Xn)
        #Ynp = Zn.data.max(dim=1)[1]
        #---------------------------        
        #0.5: https://arxiv.org/pdf/1412.6572.pdf
        loss1 = 0.5*nnF.cross_entropy(Z, Y) + 0.5*nnF.cross_entropy(Zn, Y)
        loss1.backward()             
        L1_weight_decay(optimizer, 0.01)
        L2_weight_decay(optimizer, 0.01)
        optimizer.step()
        model.update_WoW()        
        #observe these loss functions
        optimizer.step()
        model.update_WoW()

                
        loss1_train+=loss1.item()

        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss1: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), 
                   loss1.item()))
    loss1_train/=len(dataloader)
    acc_train/=sample_count
    return (loss1_train), acc_train
#%%
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
#%% ------ use this line, and then this file can be used as a Python module --------------------
def my_main(adv_noise):
    train_arg={}
    train_arg['adv']=True
    train_arg['noise']=adv_noise
    train_arg['lr']=0.001
    train_arg['device'] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    main(batch_size=128, num=128, bias=True, loss_name='ce_adv_10pgd'+str(adv_noise),
         epoch_start=50, epoch_end=50, train=train, train_arg=train_arg, evaluate_model=True)
if __name__=='__main__':
    noises = [0.1,0.2,0.3]
    #noises=[0.2,0.3]
    for n in noises:
        my_main(n)

