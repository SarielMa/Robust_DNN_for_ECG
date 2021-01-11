import sys
sys.path.append('../../core')
#%%
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim

from CPSC2018_Dataset import get_dataloader
from CPSC2018_CNN import main, main_evaluate_rand, get_filename
from RobustDNN_loss_mask import dZdX_NSR as loss2_term 
from RobustDNN_loss_mask import multi_margin_loss

random_seed = 0
#%%
#https://pytorch.org/docs/stable/notes/randomness.html
#https://pytorch.org/docs/stable/cuda.html
import random
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(random_seed)

#%%
def update_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr']=new_lr
        print('new lr=', g['lr'])
#%%
def train(model, device, optimizer, dataloader, epoch, train_arg):
    model.train()
    beta1=train_arg['beta1']
    beta2=train_arg['beta2']
    epoch_adv = train_arg['epoch_adv']
    loss1_train=0
    loss2_train=0
    loss3_train=0

    acc_train =0
    sample_count=0
    
    M = torch.tensor([[1,0,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0,0],
                      [0,0,0,0,1,0,0,0,0],
                      [0,0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,0,0,1],], device=device, dtype=torch.float32)
    
    for batch_idx, batch_data in enumerate(dataloader):
        X, Y = batch_data[0].to(device), batch_data[1].to(device)
        mask = batch_data[2].to(device)
        #Idx = batch_data[3].to(device)
        #----------------------------
        model.zero_grad()
        Z = model(X, mask)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y = Yp==Y
        #---
        MM=M[Y]
        loss1 = torch.mean(torch.sum((Z-MM)**2, dim=1))
        #loss1.backward(retain_graph=True)# MSE
        #loss1=nnF.cross_entropy(Z, Y)
        #loss1.backward(retain_graph=True)
        loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
        loss3=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
        if epoch >= epoch_adv:
            loss2 = multi_margin_loss(Z[Yp_e_Y], Y[Yp_e_Y], margin=1, within_margin=True, reduction='counter')
            #(beta1*loss2).backward(retain_graph=True)# Multi margin loss
            loss3 = loss2_term(model, X[Yp_e_Y], Y[Yp_e_Y], mask = mask[Yp_e_Y], log_NSR=True, eps=0.01)
        loss = loss1+(beta1*loss2)+(beta2*loss3)
        loss.backward()# compute the gredient
        optimizer.step()#update
        #----------------------------
        #---------------------------
        loss1_train+=loss1.detach().item()
        loss2_train+=loss2.detach().item()
        loss3_train+=loss3.detach().item()
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.detach().size(0)

        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), loss1.item(), loss2.item(), loss3.item()))
    #---------------------------
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)

    acc_train/=sample_count
    #---------------------------
    return (loss1_train, loss2_train, loss3_train), acc_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--norm_type', default=np.inf, type=float)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epoch_adv', default=10, type=int)
    parser.add_argument('--epoch_end', default=70, type=int)
    parser.add_argument('--beta1', default=1, type=float)
    parser.add_argument('--beta2', default=1.0, type=float)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--DataParallel', default=0, type=int)
    parser.add_argument('--device_ids', default=[0,1,2,3], nargs='+', type=int)
    parser.add_argument('--rand_pad', default=True, type=int)
    parser.add_argument('--net_name', default='resnet18a', type=str)


    arg = parser.parse_args()
    print(arg)
    #-------------------------------------------
    #sample_count_train = 5905
    device=torch.device('cuda:'+str(arg.cuda_id) if torch.cuda.is_available() else "cpu")
    arg=vars(arg)
    for b2 in [1.0]:
        arg['beta2'] = b2
        loss_name=(arg['net_name']+str(arg['beta2'])+'LossDZDXNSR_10adv_'+str(arg['beta1'])+'MarginLoss'
                   +'_'+arg['optimizer']
                   +'_'+str(arg['rand_pad']))
    
        if random_seed >0:
            loss_name+='_rs'+str(random_seed)
        #-------------------------------------------
        
        arg['loss_name']=loss_name
        arg['device'] = device
        print ("This beta2 is ", arg['beta2'])
        main(epoch_start=arg['epoch_start'], epoch_end=arg['epoch_end'], train=train, arg=arg, evaluate_model=True)
    #%%
    
    #if 0:
        #%%
        
        loader_train, loader_val, loader_test = get_dataloader()
        main_evaluate_rand(net_name=arg['net_name'],
                           loss_name=loss_name,
                           epoch=int (arg['epoch_end'])-1,
                           device=torch.device('cuda:'+str(arg['cuda_id']) if torch.cuda.is_available() else "cpu"),
                           loader=loader_test,
                           noise_norm_list=(0.05, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6))
