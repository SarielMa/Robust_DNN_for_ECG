import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from RobustDNN_loss import dZdX_loss1cor_obsv, dZdX_loss3zs, multi_margin_loss
from ECG_Dataset import get_dataloader, get_dataloader_bba
from ECG_CNN1 import Net, main, get_filename, main_evaluate_bba_spsa
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
def train(model, device, optimizer, dataloader, epoch, train_arg):
    model.train()#set model to training mode
    beta=train_arg['beta']
    loss1_train=0
    loss2_train=0
    loss3_train=0
    loss4_train=0
    acc_train =0
    sample_count=0
    M = torch.tensor([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0],
                      [0,0,0,1,0],
                      [0,0,0,0,1]], device=device, dtype=torch.float32)
    if epoch > 0:
        model.initialize_dead_kernel()
    model.zero_WoW()
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #---------------------------
        model.zero_grad()
        model.normalize_kernel()
        Z = model(X)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y = Yp==Y
        #---------------------------
        MM=M[Y]
        loss1 = torch.mean(torch.sum((Z-MM)**2, dim=1))
        loss1.backward(retain_graph=True)
        loss2 = multi_margin_loss(Z[Yp_e_Y], Y[Yp_e_Y], margin=1, within_margin=True, reduction='counter')
        loss2.backward(retain_graph=True)
        loss3 = dZdX_loss3zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True, eps=0.01)
        (beta*loss3).backward(retain_graph=True)
        optimizer.step()
        model.update_WoW()
        #---------------------------
        #observe
        loss4 = dZdX_loss1cor_obsv(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5)
        #---------------------------
        loss1_train+=loss1.item()
        loss2_train+=loss2.item()
        loss3_train+=loss3.item()
        loss4_train+=loss4.item()
        acc_train+= torch.sum(Yp_e_Y).item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}\tLoss4: {:.6f}'.format(
                  epoch, 100. * batch_idx / len(dataloader), loss1.item(), loss2.item(), loss3.item(), loss4.item()))

    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)
    loss4_train/=len(dataloader)
    acc_train/=sample_count
    return (loss1_train, loss2_train, loss3_train, loss4_train), acc_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
#%%    
    #beta=0.5
    for b in [0.3]:
        train_arg={}
        train_arg['lr']=0.001
        train_arg['beta']=b
        
        train_arg['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        main(batch_size=128,num=128, bias=True, loss_name='mse_margin1_loss3zs_beta'+str(b),
             epoch_start=50, epoch_end=50, train=train, train_arg=train_arg, evaluate_model=True)
#%%
    """    
    from Evaluate_bba_spsa import test_adv
    device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    filename=get_filename(num=128, bias=True, loss_name='mse_margin1_0.5loss3zs', epoch=49)
    checkpoint=torch.load(filename+'.pt', map_location=device)
    model=Net(num=128, bias=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()        
    loader_bba = get_dataloader_bba()            
    result10=test_adv(model, device, loader_bba, 5, 0.1, method='spsa_attack2',spsa_samples=2048)
    #result20=test_adv(model, device, loader_bba, 5, 0.2, method='spsa',spsa_samples=2048)
    #result30=test_adv(model, device, loader_bba, 5, 0.3, method='spsa',spsa_samples=2048)    
#%%
    device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    loader_bba = get_dataloader_bba()
    main_evaluate_bba_spsa(num=128, bias=True, loss_name='mse_margin1_0.5loss3zs',
                           epoch=49, device=device, loader=loader_bba)
    """
#%%
'''
testing robustness, noise_norm= 0.1 , max_iter= 100 , step= 0.01
acc_clean 0.9185185185185185 , acc_noisy 0.519753086419753
sens_clean [0.98148148 0.82098765 0.95061728 0.88271605 0.95679012]
sens_noisy [0.91975309 0.4382716  0.51851852 0.16049383 0.5617284 ]
prec_clean [0.76811594 0.98518519 0.93902439 0.95973154 1.        ]
prec_noisy [0.32461874 0.8875     0.63157895 0.57777778 0.97849462]   
'''