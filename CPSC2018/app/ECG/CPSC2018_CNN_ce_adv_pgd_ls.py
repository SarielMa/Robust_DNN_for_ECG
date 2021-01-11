import sys
sys.path.append('../../core')
#%%
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn.functional as nnF
from Evaluate_mask import pgd_attack_original as pgd_attack
from CPSC2018_Dataset import get_dataloader
from CPSC2018_CNN import main, main_evaluate_rand
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
    model.train()
    noise=train_arg['noise']
    epoch_adv=train_arg['epoch_adv']
    epoch_end=train_arg['epoch_end']
    if epoch >= epoch_adv:
        noise=noise*(epoch-epoch_adv+1)/(epoch_end-epoch_adv)
    norm_type=train_arg['norm_type']
    pgd_loss_fn=train_arg['pgd_loss_fn']
    max_iter=train_arg['max_iter']
    alpha=train_arg['alpha']
    step=alpha*noise/max_iter
    beta=train_arg['beta']
    print('epoch', epoch, 'noise', noise, 'step', step, 'norm_type', norm_type, 'beta', beta)
    loss_train=0
    loss1_train=0
    loss2_train=0
    acc1_train =0
    acc2_train =0
    sample_count=0
    for batch_idx, batch_data in enumerate(dataloader):
        X, Y = batch_data[0].to(device), batch_data[1].to(device)
        Mask = batch_data[2].to(device)
        #---------------------------        
        Z = model(X, Mask)
        Yp = Z.data.max(dim=1)[1]
        Yp_e_Y=Yp==Y
        loss1=nnF.cross_entropy(Z, Y, reduction='mean')
        loss1_train+=loss1.item()
        acc1_train+= torch.sum(Yp_e_Y).item()
        loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
        if Yp_e_Y.sum().item()>0 and epoch >= epoch_adv:
            Yn = Y[Yp_e_Y]
            Maskn=Mask[Yp_e_Y]
            Xn = pgd_attack(model, X[Yp_e_Y], Yn, Maskn, noise_norm=noise, norm_type=norm_type,
                            max_iter=max_iter, step=step, loss_fn=pgd_loss_fn,
                            clip_X_min=-1, clip_X_max=1)            
            Zn = model(Xn, Maskn)
            Ypn = Zn.data.max(dim=1)[1]
            loss2=nnF.cross_entropy(Zn, Yn, reduction='sum')/X.size(0)
            loss2_train+=loss2.item()
            acc2_train+= torch.sum(Ypn==Yn).item()
        #---------------------------
        loss = (1-beta)*loss1 + beta*loss2
        model.zero_grad()
        loss.backward()
        optimizer.step()
        #---------------------------
        loss_train+=loss.item()
        sample_count+=X.size(0)
        if batch_idx % 50 == 0:
            print('''Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}\tLoss1: {:.6f}\tLoss2: {:.6f}'''.format(
                   epoch, 100. * batch_idx / len(dataloader), loss.item(),loss1.item(),loss2.item()))
    loss_train/=len(dataloader)
    loss1_train/=len(dataloader)
    loss2_train/=len(dataloader)
    acc1_train/=sample_count
    acc2_train/=sample_count
    return (loss_train, loss1_train, loss2_train), (acc1_train, acc2_train)
#%% ------ use this line, and then this file can be used as a Python module --------------------
if __name__ == '__main__':
#%%
    parser = argparse.ArgumentParser(description='Input Parameters:')
    parser.add_argument('--cuda_id', default=0, type=int)
    parser.add_argument('--epoch_start', default=70, type=int)
    parser.add_argument('--epoch_adv', default=10, type=int)
    parser.add_argument('--epoch_end', default=70, type=int)
    parser.add_argument('--pgd_loss_fn', default='slm', type=str)
    parser.add_argument('--max_iter', default=20, type=int)
    parser.add_argument('--alpha', default=4, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--rand_pad', default=True, type=bool)
    parser.add_argument('--net_name', default='resnet18a', type=str)
    arg = parser.parse_args()
    arg=vars(arg)
    print(arg)
    for norm_type in [np.inf]:
        if norm_type == np.inf:
            noise_list=[0.01, 0.05, 0.1]
        for noise in noise_list:
            loss_name=('ce_'+str(arg['beta'])+'advls'+str(arg['epoch_end'])
                       +'_'+str(arg['max_iter'])+'a'+str(arg['alpha'])+'pgd'+str(noise)+'L'+str(norm_type)
                       +'_'+arg['pgd_loss_fn']
                       +'_'+arg['optimizer']
                       +'_rp'+str(arg['rand_pad'])[0])
            arg['noise']=noise
            arg['norm_type']=norm_type
            arg['device'] = torch.device("cuda:"+str(arg['cuda_id']) if torch.cuda.is_available() else "cpu")
            arg['loss_name']=loss_name
            main(epoch_start=arg['epoch_start'], epoch_end=arg['epoch_end'],
                 train=train, arg=arg, evaluate_model=True)
#%%
            """loader_train, loader_val, loader_test = get_dataloader()
            main_evaluate_rand(net_name=arg['net_name'],
                               loss_name=loss_name,
                               epoch=int (arg['epoch_end'])-1,
                               device=torch.device('cuda:'+str(arg['cuda_id']) if torch.cuda.is_available() else "cpu"),
                               loader=loader_test,
                               noise_norm_list=(0.05, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6))"""