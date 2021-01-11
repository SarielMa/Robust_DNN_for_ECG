import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
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



def plot_result(list_accs,list_precs,list_labels,noise_levels, suffix):
    #fig, ax = plt.subplots(1, 3)
    colors = ['-b','-g','-r','-y','-k','-m','-c']
    #markers = [".","^"]
    linesty = ["-","-."]
    outpath1 ="Result_for_paper/ACC"+suffix
    outpath2 ="Result_for_paper/PREC"+suffix
    outpath = "Result_for_paper/"+suffix

    #plt.figure(figsize=(10,5))
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,6), sharey=True)


    init_mk1 = 0
    mk1 = init_mk1
    for i,_ in enumerate(list_labels):
        j = i
        if j > 5:
            mk1 = (init_mk1 + int(j/5))
            j = j%5
        ax1.plot(list_accs[i], colors[j], linestyle = linesty[mk1], label = list_labels[i])

    ax1.set_xlabel('noise_level')
    ax1.set_ylabel('acc')
    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.05, step=0.1))
    positions = list(range(len(noise_levels)))
    labels = tuple([str(i) for i in noise_levels])
    ax1.set_xticks(positions, labels)


    init_mk2 = 0
    mk2 = init_mk2
    for i,_ in enumerate(list_labels):
        j = i
        if j > 5:
            mk2 = (init_mk2 + int(j/5))
            j = j%5
        ax2.plot(list_accs[i], colors[j], linestyle = linesty[mk2], label = list_labels[i])

    ax2.set_xlabel('noise_level')
    ax2.set_ylabel('prec')
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.arange(0, 1.05, step=0.1))
    positions = list(range(len(noise_levels)))
    labels = tuple([str(i) for i in noise_levels])
    ax2.set_xticks(positions, labels)


    plt.legend()
    plt.grid(True)
    plt.savefig(outpath+".png")
    plt.close()

    with open(outpath1+".csv",'w') as csvf:
        fw = csv.writer(csvf)
        fw.writerow(["noise level"]+noise_levels)
        for i,l in enumerate(list_labels):
            fw.writerow([l]+list_accs[i])

    with open(outpath2+".csv",'w') as csvf:
        fw = csv.writer(csvf)
        fw.writerow(["noise level"]+noise_levels)
        for i,l in enumerate(list_labels):
            fw.writerow([l]+list_precs[i])

def plot_result_prec(list_precs,list_labels,noise_levels, suffix):
    #fig, ax = plt.subplots(1, 3)
    colors = ['-b','-g','-r','-y','-k','-m','-c']
    markers = [".","^"]
    linesty = ["-","-."]
    #outpath1 ="Result_for_paper/ACC"+suffix
    outpath2 ="Result_for_paper/PREC"+suffix

    init_mk = 0
    mk = init_mk
    plt.figure(figsize=(6,6))
    for i,_ in enumerate(list_labels):
        j = i
        if j > 5:
            mk = (init_mk + int(j/5))
            j = j%5
        plt.plot(list_precs[i], colors[j], linestyle = linesty[mk], label = list_labels[i])

    plt.xlabel('noise_level')
    plt.ylabel('PREC')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.05, step=0.05))

    positions = list(range(len(noise_levels)))
    labels = tuple([str(i) for i in noise_levels])
    plt.xticks(positions, labels)
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath2+".svg",dpi=600)
    plt.close()

    with open(outpath2+".csv",'w') as csvf:
        fw = csv.writer(csvf)
        fw.writerow(["noise level"]+noise_levels)
        for i,l in enumerate(list_labels):
            fw.writerow([l]+list_precs[i])

def plot_result_acc(list_acc,list_labels,noise_levels, suffix):
    #fig, ax = plt.subplots(1, 3)
    colors = ['-b','-g','-r','-y','-k','-m','-c']
    markers = [".","^"]
    linesty = ["-","-."]
    #outpath1 ="Result_for_paper/ACC"+suffix
    outpath2 ="Result_for_paper/ACC"+suffix

    init_mk = 0
    mk = init_mk
    plt.figure(figsize=(6,8))
    for i,_ in enumerate(list_labels):
        j = i
        if j > 5:
            mk = (init_mk + int(j/5))
            j = j%5
        plt.plot(list_acc[i], colors[j], linestyle = linesty[mk], label = list_labels[i])

    plt.xlabel('noise_level')
    plt.ylabel('ACC')
    #plt.figure(figsize=(6,8))
    #plt.ylim(0, 1)
    #plt.yticks(np.arange(0, 1.05, step=0.1))

    positions = list(range(len(noise_levels)))
    labels = tuple([str(i) for i in noise_levels])
    plt.xticks(positions, labels)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.05, step=0.05))
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath2+".svg",dpi=600,bbox_inches='tight',pad_inches=0.0)
    plt.close()

    with open(outpath2+".csv",'w') as csvf:
        fw = csv.writer(csvf)
        fw.writerow(["noise level"]+noise_levels)
        for i,l in enumerate(list_labels):
            fw.writerow([l]+list_acc[i])

#%% ------ use this line, and then this file can be used as a Python module --------------------
def main(nt,at):
    #epoch_save = 49
    nettype = ["CNN1_",
               "MLP1_"]
    losses = ["128True_ce_epoch49_",
            "128True_ce_Jacob1__epoch49_",
            "128True_mse_epoch49_",
            "128True_mse_margin_epoch49_",
            "128True_mse_margin_loss3zs_epoch49_",
            "128False_mse_margin_loss1_epoch49_",
            "128True_ce_adv_10pgd0.1_epoch49_",
            "128True_ce_adv_10pgd0.2_epoch49_",
            "128True_ce_adv_10pgd0.3_epoch49_"
            ]

    tails = ["result_wba.pt","result_bba_spsa.pt"]

    tests = ["result_20pgd" , "result_100pgd","result_spsa"]


    #test = tests[at]

    #for this_test in tests:
    this_test = tests[at]
    suffix = nettype[nt].split("_")[0] + this_test.split("_")[1]
    list_labels=["ce","jacob","mse","mseMargin","NSR","loss1","adv0.1","adv0.2","adv0.3"]
    loss_list = [0,1,2,3,4,5,6,7,8]
    noise_levels = [0,0.01, 0.03, 0.05, 0.1, 0.2, 0.3]


    list_accs=[]
    list_precs=[]
    for m in loss_list:#for each methods
        if at in [0,1]:
            fname = "ECG_"+nettype[nt]+losses[m]+tails[0]
        else:
            fname = "ECG_"+nettype[nt]+losses[m]+tails[1]
        print (fname)
        checkpoint1=torch.load("Result_for_paper_old/"+fname)

        results=checkpoint1[this_test]

        accs = []
        precs = []
        accs.append(round(results[0]['acc_clean'], 2))
        precs.append(round(np.nanmean(results[0]['prec_clean']),2))
        for i in range(len(results)):
            if at!=1:
                if i in [4,6]:
                    continue
            accs.append(round(results[i]['acc_noisy'],2))
            precs.append(round(np.nanmean(results[i]['prec_noisy']),2))

        list_accs.append(accs)
        list_precs.append(precs)

    print (list_precs)
    #plot_result(list_accs,list_precs,list_labels,noise_levels,suffix)
    plot_result_acc(list_accs,list_labels,noise_levels,suffix)
    plot_result_prec(list_precs,list_labels,noise_levels,suffix)

    



if __name__=='__main__':
    for i in [0,1]:
        for j in [0,1,2]:
            main(i,j)