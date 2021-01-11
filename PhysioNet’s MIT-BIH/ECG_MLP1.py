import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from RobustDNN_module import Linear
from ECG_Dataset import get_dataloader, get_dataloader_bba
from Evaluate import test_adv, test_rand
from Evaluate_bba1 import test_adv as test_adv_bba1
from Evaluate_bba1 import train_sub_model
from Evaluate_bba_spsa import test_adv as test_adv_spsa
#%% Net: fully connected feedforward Neural Network
class Net(nn.Module):
    def __init__(self, num, bias):
        super().__init__()

        self.E = nn.ModuleList([Linear(187, num),
                                Linear(num, num),
                                Linear(num, num),
                                Linear(num, 32)])

        self.C = nn.ModuleList([Linear(32, 5)])

        self.G = nn.ModuleList([nn.ReLU(inplace=True),
                                nn.ReLU(inplace=True),
                                nn.ReLU(inplace=True)])
        if bias == False:
            self.disable_bias()
        
    def disable_bias(self):
        for n in range(0, len(self.E)):
            self.E[n].disable_bias()
        self.C[0].disable_bias()
            
    def normalize_kernel(self):
        for n in range(0, len(self.E)):
            self.E[n].normalize_kernel()

    def zero_WoW(self):
        for n in range(0, len(self.E)):
            self.E[n].zero_WoW()

    def update_WoW(self):
        for n in range(0, len(self.E)):
            self.E[n].update_WoW()

    def initialize_dead_kernel(self):
        for n in range(0, len(self.E)):
            counter=self.E[n].initialize_dead_kernel()
            print('initialize_dead_kernel in E, counter=', counter)

    def forward(self, x):
        #self.feature_in=[]
        #self.feature_out=[]
        for n in range(0, len(self.E)):
            #self.feature_in.append(x)
            x=self.E[n](x)
            #self.feature_out.append(x)
            if n < len(self.E)-1:
                x=self.G[n](x)
        z=self.C[0](x)
        return z    
#%%
class SubstituteNet(nn.Module):
    def __init__(self, num, bias):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(187, num, bias=bias),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(num, affine=False),
                                 nn.Linear(num, num, bias=bias),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(num, affine=False),
                                 nn.Linear(num, num, bias=bias),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(num, affine=False),
                                 nn.Linear(num, num, bias=bias),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(num, affine=False),
                                 nn.Linear(num, 5, bias=bias))
    def forward(self, x):
        z=self.net(x)
        return z    
#%%
def save_checkpoint(filename, model, optimizer, result, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)
#%%
def plot_result(loss_train_list, acc_train_list,
                loss_val_list, acc_val_list,
                loss_test_list, acc_test_list,
                adv_acc_val_list, adv_acc_test_list):
    fig, ax = plt.subplots(1, 3, figsize=(9,3))
    ax[0].set_title('loss v.s. epoch')
    ax[0].plot(loss_train_list, '-b', label='training loss')
    ax[0].plot(loss_val_list, '-g', label='validation loss')
    ax[0].plot(loss_test_list, '-r', label='testing loss')
    ax[0].set_xlabel('epoch')
    #ax[0].set_xticks(np.arange(len(loss_train_list)))
    #ax[0].legend()
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch')
    ax[1].plot(acc_train_list, '-b', label='training acc')
    ax[1].plot(acc_val_list, '-g', label='validation acc')
    ax[1].plot(acc_test_list, '-r', label='testing acc')
    ax[1].set_xlabel('epoch')
    #ax[1].legend()
    ax[1].grid(True)
    ax[2].set_title('accuracy v.s. epoch')
    ax[2].plot(adv_acc_val_list, '-c', label='adv val acc')
    ax[2].plot(adv_acc_test_list, '-m', label='adv test acc')
    ax[2].set_xlabel('epoch')
    #ax[2].legend()
    ax[2].grid(True)
    return fig, ax    
#%%
def get_filename(num, bias, loss_name, train_arg, epoch=None):
    filename='result/ECG_MLP1_'+str(num)+str(bias)+'_'+loss_name

    if 'jacob1' in train_arg:
        filename =filename+'_Jacob1_'
    elif 'jacob2' in train_arg:
        filename =filename+'_Jacob2_'
    if epoch is not None:
        filename =filename+'_epoch'+str(epoch)
    print ("filename is ",filename)
    return filename
#%%
def main(batch_size, num, bias, loss_name, epoch_start, epoch_end, train, train_arg, evaluate_model):
    main_train(batch_size,num, bias, loss_name, epoch_start, epoch_end, train, train_arg)
    if evaluate_model == True:
        epoch_save=epoch_end-1
        device=train_arg['device']
        loader_train, loader_val, loader_test = get_dataloader(batch_size)
        del loader_train, loader_val
        print ("only bba")
        #main_evaluate_wba(train_arg, num, bias, loss_name, epoch_save, device, loader_test)
        #main_evaluate_bba(train_arg, num, bias, loss_name, epoch_save, device, loader_test)
        #main_evaluate_rand(train_arg,num, bias, loss_name, epoch_save, device, loader_test)
        main_evaluate_SAP(train_arg,num, bias, loss_name, epoch_save, device, loader_test)
        #-------------------------------------
        #loader_bba = get_dataloader_bba()
        #main_evaluate_bba_spsa(train_arg,num, bias, loss_name, epoch_save, device, loader_bba)
#%%
def main_train(batch_size,num, bias, loss_name, epoch_start, epoch_end, train, train_arg):
#%%
    filename=get_filename(num, bias, loss_name, train_arg)
    print('train model: '+filename)
    lr=train_arg['lr']
    device=train_arg['device']
#%%
    loader_train, loader_val, loader_test = get_dataloader(batch_size)
#%%
    model=Net(num, bias)    
    model.to(device)
    #%%
    x=loader_train.dataset[0][0]
    x=x.view(1,187).to(device)
    z=model(x)
#%%
    optimizer = optim.Adamax(model.parameters(), lr=lr)
#%%
    loss_train_list=[]
    acc_train_list=[]
    loss_val_list=[]
    acc_val_list=[]
    loss_test_list=[]
    acc_test_list=[]
    adv_acc_val_list=[]
    adv_acc_test_list=[]
    epoch_save=epoch_start-1
#%%
    if epoch_start>0:
        if epoch_save >= 0:
            checkpoint=torch.load(filename+'_epoch'+str(epoch_save)+'.pt', map_location=device)
            model=Net(num, bias)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            (loss_train_list, acc_train_list,
             loss_val_list, acc_val_list,
             loss_test_list, acc_test_list,
             adv_acc_val_list,adv_acc_test_list)=checkpoint['result']
            optimizer = optim.Adamax(model.parameters(), lr=lr)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])      
#%%
    for epoch in range(epoch_save+1, epoch_end):
        #-------- training --------------------------------
        loss_train, acc_train =train(model, device, optimizer, loader_train, epoch, train_arg)
        loss_train_list.append(loss_train)
        acc_train_list.append(acc_train)
        print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
        #-------- validation --------------------------------
        result_val = test_adv(model, device, loader_val, 5, 0.1)
        #loss_val_list.append(result_val['loss_ce'])
        acc_val_list.append(result_val['acc_clean'])
        adv_acc_val_list.append(result_val['acc_noisy'])
        #-------- testing --------------------------------
        result_test = test_adv(model, device, loader_test, 5, 0.1)
        #loss_test_list.append(result_test['loss_ce'])
        acc_test_list.append(result_test['acc_clean'])
        adv_acc_test_list.append(result_test['acc_noisy'])
        #--------save model-------------------------
        result = (loss_train_list, acc_train_list,
                  loss_val_list, acc_val_list,
                  loss_test_list, acc_test_list,
                  adv_acc_val_list, adv_acc_test_list)
        save_checkpoint(filename+'_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
        epoch_save=epoch
        #------- show result ----------------------
        display.clear_output(wait=False)
        plt.close('all')
        fig, ax = plot_result(loss_train_list, acc_train_list,
                              loss_val_list, acc_val_list,
                              loss_test_list, acc_test_list,
                              adv_acc_val_list, adv_acc_test_list)
        display.display(fig)
        fig.savefig(filename+'_epoch'+str(epoch)+'.png')     
#%% 
def main_evaluate_wba(train_arg, num, bias, loss_name, epoch, device, loader):
#%%
    filename=get_filename(num, bias, loss_name, train_arg, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=device)
    model=Net(num, bias)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')
    
#%% ifgsm and stop_if_success   
    """
    result_ifgsm=[]
    for noise_norm in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        result_ifgsm.append(test_adv(model, device, loader, 5, noise_norm))
    fig, ax = plt.subplots()
    noise=[0]
    acc=[result_ifgsm[0]['acc_clean']]
    for k in range(0, len(result_ifgsm)):
        noise.append(result_ifgsm[k]['noise_norm'])
        acc.append(result_ifgsm[k]['acc_noisy'])
        ax.plot(noise, acc, '.-b')
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.05, step=0.05))
        ax.grid(True)
        ax.set_title('wba_ifgsm')
        ax.set_xlabel(filename)
        display.display(fig)
        fig.savefig(filename+'_wba_ifgsm.png')
    """
#%% 10pgd and stop_if_success  
    """
    result_10pgd=[]
    for noise_norm in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        result_10pgd.append(test_adv(model, device, loader, 5, noise_norm, 10, 0.01, method='pgd'))
    fig, ax = plt.subplots()
    noise=[0]
    acc=[result_10pgd[0]['acc_clean']]
    for k in range(0, len(result_10pgd)):
        noise.append(result_10pgd[k]['noise_norm'])
        acc.append(result_10pgd[k]['acc_noisy'])
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    ax.set_title('wba_10pgd')
    ax.set_xlabel(filename)
    display.display(fig)
    fig.savefig(filename+'_wba_10pgd.png')      
#%% 20pgd and stop_if_success
    result_20pgd=[]
    for noise_norm in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        result_20pgd.append(test_adv(model, device, loader, 5, noise_norm, 20, 0.01, method='pgd'))
    fig, ax = plt.subplots()
    noise=[0]
    acc=[result_20pgd[0]['acc_clean']]
    for k in range(0, len(result_20pgd)):
        noise.append(result_20pgd[k]['noise_norm'])
        acc.append(result_20pgd[k]['acc_noisy'])
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    ax.set_title('wba_20pgd')
    ax.set_xlabel(filename)
    display.display(fig)
    fig.savefig(filename+'_wba_20pgd.png')  
    """
#%% 100pgd and stop_if_success
    result_100pgd=[]
    for noise_norm in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        result_100pgd.append(test_adv(model, device, loader, 5, noise_norm, 100, 0.01, method='pgd'))
    fig, ax = plt.subplots()
    noise=[0]
    acc=[result_100pgd[0]['acc_clean']]
    for k in range(0, len(result_100pgd)):
        noise.append(result_100pgd[k]['noise_norm'])
        acc.append(result_100pgd[k]['acc_noisy'])
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    ax.set_title('wba_100pgd')
    ax.set_xlabel(filename)
    display.display(fig)
    fig.savefig(filename+'_wba_100pgd.png')
#%%  
    torch.save({'result_100pgd':result_100pgd},
               filename+'_result_wba.pt')
#%%
def main_evaluate_SAP(train_arg, num, bias, loss_name, epoch, device, loader):
    #https://www.nature.com/articles/s41591-020-0791-x
    filename=get_filename(num, bias, loss_name, train_arg, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=device)
    model=Net(num, bias)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_wba model in '+filename+'.pt')

    result_100pgd=[]
    max_iter = 100
    for noise_norm in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        result_100pgd.append(test_adv(model, device, loader, 5, noise_norm, max_iter, 0.01, method='SAP'))
    fig, ax = plt.subplots()
    noise=[0]
    acc=[result_100pgd[0]['acc_clean']]
    for k in range(0, len(result_100pgd)):
        noise.append(result_100pgd[k]['noise_norm'])
        acc.append(result_100pgd[k]['acc_noisy'])
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    ax.set_title('wba_'+str(max_iter)+'sap')
    ax.set_xlabel(filename)
    display.display(fig)
    fig.savefig(filename+'_wba_'+str(max_iter)+'sap.png')
 
    torch.save({'result_'+str(max_iter)+'sap':result_100pgd},
               filename+'_result_'+str(max_iter)+'sap.pt')
#%%
def main_evaluate_bba_spsa(train_arg,num, bias, loss_name, epoch, device, loader):

    filename=get_filename(num, bias, loss_name, train_arg, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=device)
    model=Net(num, bias)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_bba_spsa model in '+filename+'.pt')
#%%
    result_spsa=[]
    for noise_norm in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        result_spsa.append(test_adv_spsa(model, device, loader, 5, noise_norm, method='spsa_attack',spsa_samples=2048))
        fig, ax = plt.subplots()
    noise=[0]
    acc=[result_spsa[0]['acc_clean']]
    for k in range(0, len(result_spsa)):
        noise.append(result_spsa[k]['noise_norm'])
        acc.append(result_spsa[k]['acc_noisy'])
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    ax.set_title('bba_spsa')
    ax.set_xlabel(filename)
    display.display(fig)
    fig.savefig(filename+'_bba_spsa.png')            
    #------------------------------------
    torch.save({'result_spsa':result_spsa},filename+'_result_bba_spsa.pt')
def main_evaluate_rand(train_arg,num, bias, loss_name, epoch, device, loader):
#%%
    filename=get_filename(num, bias, loss_name, train_arg, epoch)
    checkpoint=torch.load(filename+'.pt', map_location=device)
    model=Net(num, bias)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('evaluate_rand model in '+filename+'.pt')
#%%
    result_rand=[]
    for noise_norm in [0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        result_rand.append(test_rand(model, device, loader, 5, noise_norm))
        fig, ax = plt.subplots()
    noise=[]
    acc=[]
    for k in range(0, len(result_rand)):
        noise.append(result_rand[k]['noise_norm'])
        acc.append(result_rand[k]['acc_noisy'])
    ax.plot(noise, acc, '.-b')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    ax.set_title('rand')
    ax.set_xlabel(filename)
    display.display(fig)
    fig.savefig(filename+'_rand.png')            
    #------------------------------------
    torch.save({'result_rand':result_rand},filename+'_result_rand.pt')