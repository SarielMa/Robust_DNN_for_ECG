import sys
import os
sys.path.append('../../core')
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torchvision.models as tv_models
import math
import csv


def cal_AUC_robustness(acc_list, noise_level_list, maxNoise):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
        if noise_level_list[n]==maxNoise:
            break
    auc/=noise_level_list[n]
    return auc

def F1(confusion):
    num_classes = 5
    F1s = np.zeros(num_classes)
    for n in range(0, num_classes):
        F1s[n] = (confusion[n,n]*2)/(np.sum(confusion[n,:])+np.sum(confusion[:,n]))
    return np.mean(F1s)


def show_MLP_Jacob():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='val_result/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    ax.set_xlim(0.1)
    noise_norm_list=(0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Jacob" in fn and "wba" in fn and "MLP" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------

            

            for beta in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
                lt = "beta"+str(beta)+"_Jacob"
                if lt in fn:
                    col += 1
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    acc = [acc_all[0]['acc_clean']]
                    for line in acc_all:
                        acc.append(line['acc_noisy'])
                        
                    if col>6:
                        ax.plot(acc, colors[col%6],label = str(beta)+"Jacob", marker="o")
                    else:
                        
                        ax.plot(acc, colors[col],label = str(beta)+"Jacob")
                    accs.append(acc)
                    list_labels.append(str(beta)+"Jacob")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_Jacob_MLP.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="Jacob_MLP_validation", flag="acc" )

def show_CNN_Jacob():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    title='wba_100pgd_L'+str(norm_type)
    pre_fix='val_result/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    ax.set_xlim(0.1)
    noise_norm_list=(0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "Jacob" in fn and "wba" in fn and "CNN" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------

            

            for beta in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
                lt = "beta"+str(beta)+"_Jacob"
                if lt in fn:
                    col += 1
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    acc = [acc_all[0]['acc_clean']]
                    for line in acc_all:
                        acc.append(line['acc_noisy'])
                        
                    if col>6:
                        ax.plot(acc, colors[col%6],label = str(beta)+"Jacob", marker="o")
                    else:
                        
                        ax.plot(acc, colors[col],label = str(beta)+"Jacob")
                    accs.append(acc)
                    list_labels.append(str(beta)+"Jacob")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_Jacob_CNN.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="Jacob_CNN_validation", flag="acc" )

def show_CNN_loss3zs():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    pre_fix='val_result/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    ax.set_xlim(0.1)
    noise_norm_list=(0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "loss3zs" in fn and "wba" in fn and "CNN" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------

            

            for beta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                lt = "beta"+str(beta)
                if lt in fn:
                    col += 1
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    acc = [acc_all[0]['acc_clean']]
                    for line in acc_all:
                        acc.append(line['acc_noisy'])
                        
                    if col>6:
                        ax.plot(acc, colors[col%6],label = str(beta)+"NSR", marker="o")
                    else:
                        
                        ax.plot(acc, colors[col],label = str(beta)+"NSR")
                    accs.append(acc)
                    list_labels.append(str(beta)+"NSR")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_loss3zs_CNN.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="NSR_CNN_validation", flag="acc" )
    
def show_MLP_loss3zs():
    # for adv and
    fig, ax = plt.subplots(figsize=(6,8))
    #plt.figure(figsize=(6,8))
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True)
    norm_type = np.inf
    pre_fix='val_result/'
    #ax.set_title(title)
    ax.set_xlabel("Noise")  
    ax.set_ylabel("ACC")
    ax.set_xlim(0.1)
    noise_norm_list=(0.01, 0.03, 0.05, 0.1, 0.2, 0.3)
    device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    #loader_train, loader_val, loader = get_dataloader()
    col = -1
    colors = ['-b','-g','-r','-y','-k','-m','-c'] 
    accs=[]
    list_labels = []
    for fn in os.listdir(pre_fix):
        
        if "loss3zs" in fn and "wba" in fn and "MLP" in fn:
            #loss_name=('resnet18a'+str(beta)+'Loss2'+str(1)+'MarginLoss'+'_'+'Adam'+'_'+str(True))
            #filename=pre_fix+net_name+"_"+loss_name+'_epoch'+str(epoch)
            filename=pre_fix+fn
            #-------------------------------------------

            

            for beta in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                lt = "beta"+str(beta)
                if lt in fn:
                    col += 1
                    print (fn)
                    checkpoint=torch.load(filename, map_location=torch.device('cpu'))
                    acc_all = checkpoint['result_100pgd']
                    acc = [acc_all[0]['acc_clean']]
                    for line in acc_all:
                        acc.append(line['acc_noisy'])
                        
                    if col>6:
                        ax.plot(acc, colors[col%6],label = str(beta)+"NSR", marker="o")
                    else:
                        
                        ax.plot(acc, colors[col],label = str(beta)+"NSR")
                    accs.append(acc)
                    list_labels.append(str(beta)+"NSR")
                  
    positions = list(range(len(noise_norm_list)+1))
    labels = tuple([0]+[str(i) for i in noise_norm_list])
    plt.xticks(positions, labels) 
    ax.legend()
    
    fig.savefig(pre_fix+'Evaluate_loss3zs_MLP.svg',bbox_inches='tight',pad_inches=0.0)
    plt.close(fig)
    
    get_table (accs=accs, list_labels=list_labels, noise_levels=[0]+list(noise_norm_list), 
               pre_fix=pre_fix, dataset="NSR_MLP_validation", flag="acc" )
    
def get_table(accs,list_labels,noise_levels, pre_fix, dataset,  flag):
    #fig, ax = plt.subplots(1, 3)
    #colors = ['-b','-g','-r','-y','-k','-m','-c']
    #markers = ['.','s','^']
    outpath1 =pre_fix +dataset
    outpath2 =pre_fix +dataset
    
    if flag == "acc":
        with open(outpath1+".csv",'w') as csvf:
            fw = csv.writer(csvf)
            fw.writerow([""]+noise_levels+["AUC(before 0.1)"]+["sqrt(CleanACC*AUC(before 0.1))"])
            for i,l in enumerate(list_labels):
                fw.writerow([l+"(ACC)"]+accs[i]+[str(cal_AUC_robustness(accs[i], noise_levels, 0.1))]
                            +[str(math.sqrt(accs[i][0]*cal_AUC_robustness(accs[i], noise_levels, 0.1)))])
            
            #fw.writerow([l+"(spsa)"]+bba_spsa_accs[i])

if __name__ == '__main__':
    show_MLP_Jacob()
    show_CNN_Jacob()
    show_MLP_loss3zs()
    show_CNN_loss3zs()
