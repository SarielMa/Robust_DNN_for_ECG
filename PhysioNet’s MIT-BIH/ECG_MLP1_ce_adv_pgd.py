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
    beta=train_arg['beta']
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
        Xn = pgd_attack(model, X, Y, noise_norm=noise, max_iter=10, step=0.01)
        #---------------------------
        model.zero_grad()            
        Z = model(X)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y=Yp==Y
        Zn = model(Xn)
        #Ynp = Zn.data.max(dim=1)[1]
        #---------------------------        
        a=nnF.cross_entropy(Z, Y)
        b=nnF.cross_entropy(Zn, Y)
        loss1 = a + beta*b
        loss1.backward()             
        optimizer.step()
        model.update_WoW()
        #---------------------------        
        #observe these loss functions
        loss2 = multi_margin_loss(Z[Yp_e_Y], Y[Yp_e_Y], margin=1, within_margin=True, reduction='counter')        
        loss3 = dZdX_loss3zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)
        loss4 = dZdX_loss3zs_ref(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)        
        loss5 = dZdX_loss2zs(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)
        loss6 = dZdX_loss2zs_ref(model, X[Yp_e_Y], Y[Yp_e_Y], num_classes=5, log_NSR=True)
                
        loss1_train+=loss1.item()
        loss2_train+=loss2.item()
        loss3_train+=loss3.item()
        loss4_train+=loss4.item()
        loss5_train+=loss5.item()
        loss6_train+=loss6.item()
        acc_train+= torch.sum(Yp==Y).item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss1: {:.6f}\tLoss2: {:.6f}\tLoss3: {:.6f}
                  \tLoss4: {:.6f}\tLoss5: {:.6f}\tLoss6: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), 
                   loss1.item(), loss2.item(), loss3.item(),
                   loss4.item(), loss5.item(), loss6.item()))
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    loss3_train/=len(dataloader)
    loss4_train/=len(dataloader)
    loss5_train/=len(dataloader)
    loss6_train/=len(dataloader)
    acc_train/=sample_count
    return (loss1_train, loss2_train, loss3_train, loss4_train, loss5_train, loss6_train), acc_train
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
#%%
    #for beta=[0.5, 1]:
    for noise in [0.1, 0.2, 0.3]:
        train_arg={}
        train_arg['adv']=True
        train_arg['noise']=noise
        train_arg['beta']=1
        train_arg['lr']=0.001
        train_arg['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        main(batch_size=128, num=128, bias=True, loss_name='ce_adv_10pgd'+str(noise),
             epoch_start=50, epoch_end=50, train=train, train_arg=train_arg, evaluate_model=True)

