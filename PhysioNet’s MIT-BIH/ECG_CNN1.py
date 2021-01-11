import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from RobustDNN_module import Conv1d, Linear
from ECG_Dataset import get_dataloader, get_dataloader_bba
from Evaluate import test_adv, test_rand

#%% Net: From ArXiv 1805.00794
class Block(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.conv1 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=5, stride=1, padding=2, bias=bias)
        self.conv2 = Conv1d(in_channels=32, out_channels=32,
                            kernel_size=5, stride=1, padding=2, bias=bias)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
        x1=self.conv2(nnF.relu(self.conv1(x), inplace=True))
        x2=x1+x
        x3=nnF.relu(x2, inplace=True)
        x4=self.pool(x3)
        return x4

class Net(nn.Module):
    def __init__(self,num, bias):
        super().__init__()
        self.conv0 = Conv1d(in_channels=1, out_channels=32,
                            kernel_size=5, stride=1, padding=2, bias=bias)
        self.block = nn.ModuleList([Block(bias), Block(bias), Block(bias), Block(bias), Block(bias)])                                    
        self.linear1 = Linear(64, 32, bias=bias)
        self.linear2 = Linear(32, 5, bias=bias)

    def forward(self, x):
        #input size (N,C,L), C is 1
        x=x.view(x.size(0),1,x.size(1))
        x=self.conv0(x)
        x=self.block[0](x)
        x=self.block[1](x)
        x=self.block[2](x)
        x=self.block[3](x)
        x=self.block[4](x)
        #print(x.size())
        x=x.view(x.size(0),-1)
        x=nnF.relu(self.linear1(x), inplace=True)
        z=self.linear2(x)
        #y=nnF.softmax(z, dim=1)
        return z

    def normalize_kernel(self):
        self.conv0.normalize_kernel()
        for n in range(0, len(self.block)-1):
            self.block[n].conv1.normalize_kernel()
            self.block[n].conv2.normalize_kernel()
        self.linear1.normalize_kernel()
        self.linear2.normalize_kernel()
        
    def zero_WoW(self):
        self.conv0.zero_WoW()
        for n in range(0, len(self.block)-1):
            self.block[n].conv1.zero_WoW()
            self.block[n].conv2.zero_WoW()
        self.linear1.zero_WoW()
        self.linear2.zero_WoW()

    def update_WoW(self):
        self.conv0.update_WoW()
        for n in range(0, len(self.block)-1):
            self.block[n].conv1.update_WoW()
            self.block[n].conv2.update_WoW()
        self.linear1.update_WoW()
        self.linear2.update_WoW()
        
    def initialize_dead_kernel(self):
        counter=[]
        counter.append(self.conv0.initialize_dead_kernel())
        for n in range(0, len(self.block)-1):
            counter.append(self.block[n].conv1.initialize_dead_kernel())
            counter.append(self.block[n].conv2.initialize_dead_kernel())
        counter.append(self.linear1.initialize_dead_kernel())
        counter.append(self.linear2.initialize_dead_kernel())
        print('initialize_dead_kernel in E, counter=', counter)
#%%
class SubstituteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.conv12 = nn.Conv1d(32, 32, 5, padding =2)
        self.conv13 = nn.Conv1d(32, 32, 5, padding =2)
        self.maxpool = nn.MaxPool1d(5, stride=2)
        self.fc1 = nn.Linear(32*90, 32*10)#dense
        self.fc2 = nn.Linear(32*10, 32)
        self.fc3 = nn.Linear(32, 5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x=x.view(x.size(0),1,x.size(1))
        x = nnF.relu(self.conv1(x))#[n,32,183]
        x = nnF.relu(self.conv12(x))#[n,32, 183]
        x = nnF.relu(self.conv13(x))#[n,32,183]
        x = self.maxpool(x)#[n,32,90]
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = nnF.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
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
    filename='result/ECG_CNN1_'+str(num)+str(bias)+'_'+loss_name

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
        #print ("bba only")
        device=train_arg['device']
        loader_train, loader_val, loader_test = get_dataloader(batch_size)
        del loader_train, loader_val
        #main_evaluate_wba(train_arg, num, bias, loss_name, epoch_save, device, loader_test)
        #main_evaluate_bba(train_arg, num, bias, loss_name, epoch_save, device, loader_test)
        main_evaluate_rand(train_arg,num, bias, loss_name, epoch_save, device, loader_test)
        #main_evaluate_SAP(train_arg,num, bias, loss_name, epoch_save, device, loader_test)
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
    print ("sap begins")
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
#%%
def main_evaluate_bba_spsa(train_arg,num, bias, loss_name, epoch, device, loader):
#%%
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
    for noise_norm in [0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,0.6]:
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