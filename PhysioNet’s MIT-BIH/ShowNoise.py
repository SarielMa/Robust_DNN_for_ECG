import os
import numpy as np
import matplotlib.pyplot as plt
#from IPython import display
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim


from ECG_Dataset import get_dataloader
from RobustDNN_module import Linear
from Evaluate import pgd_attack

#%%

class Net(nn.Module):
    def __init__(self, num=128, bias=True):
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



def my_plot(X, noise):
    #t=np.linspace(0, 1.496, 187)
    #plt.title('ECG under noise '+str(noise))
    #print (X.shape)
    #X = X.view(X.shape[2],-1)
    x = X.cpu().numpy()
    plt.plot(x, '-r')
    #plt.show()
    path = "ECG_plots/"
    if not os.path.exists(path):
        print ("makeing new path ...")
        os.mkdir(path)
    plt.savefig(path + "MLPpgd100"+str(noise)+".png")
    plt.close()

def test_adv(model, device, dataloader, num_classes,
             noise_norm, max_iter=None, step=None, norm_type=np.inf, method='ifgsm', 
             scale_z=None, targeted=False, stop_if_label_change=True, use_optimizer=True,
             save_model_output=False):
    model.eval()#set model to evaluation mode
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    sample_count=0
    adv_sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]

    #---------------------
    #---------------------
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #------------------
        Z = model(X)#classify the 'clean' signal X
        if len(Z.size()) <= 1:
            Yp = (Z>0).to(torch.int64) #binary/sigmoid
        else:
            Yp = Z.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        if method == 'pgd':
            Xn = pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step, True, None, 
                            scale_z, targeted, stop_if_label_change,
                            use_optimizer=use_optimizer)
        else:
            raise NotImplementedError("other method is not implemented.")
        #------------------

    return Xn


if __name__=="__main__":
    loader_train, loader_val, loader_test = get_dataloader(batch_size=1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # get one need to be shown
    #x=loader_val.dataset[0][0]#[187]
    #Y=loader_val.dataset[0][1]
    x,Y = next(iter(loader_val))
    print ("class label is ", Y)

    #print ("x shape is ",x.shape )
    #print ("Y shape is ",Y.shape )
    X = x.view(x.shape[0],x.shape[-1])#[1,187]

    #print ("X shape is ",X.shape )
    Y=Y.to(device)
    X=X.to(device)
    # MLP
    checkpoint=torch.load('Result/ECG_MLP1_128True_ce_epoch49.pt', map_location=device)
    model=Net()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    #Result/ECG_MLP1_128True_ce_epoch49

    noise_levels = [0,0.01,0.02,0.03,0.05,0.1,0.2,0.3]
    for noise in noise_levels:
        Xn = pgd_attack(model, X, Y, noise, max_iter = 100, step = 0.01)
        print ("the original y is ", Y)
        Zn = model(Xn)
        Yn = Zn.data.max(dim=1)[1]
        print ("Y epsilon is ",Yn)
        n = Xn -X
        #print ("Xn is shape [2]is ", Xn.shape[2])
        X_d  = Xn.view(Xn.shape[1])#[187]
        n = n.view(n.shape[1])
        my_plot(n, str(noise)+"noise")
        my_plot(X_d, noise)



