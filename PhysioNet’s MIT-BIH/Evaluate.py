
import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
import sys
#%%


def pgd_conv(inputs, targets, model, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
    """
    paper available at https://www.nature.com/articles/s41591-020-0791-x
    code available at https://github.com/XintianHan/ADV_ECG/blob/master/create_adv_conv_train.py
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """
    MAX_SENTENCE_LENGTH=187
    criterion = nnF.cross_entropy
    #lengths = 
    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    #crafting_input = crafting_input.view(crafting_input.size(0), 1, crafting_input.size(1))
    crafting_target = torch.autograd.Variable(targets.clone())
    for i in range(num_steps):
        output = model(crafting_input)
        loss = criterion(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward()
        added = torch.sign(crafting_input.grad.data)
        step_output = crafting_input + step_alpha * added
        total_adv = step_output - inputs
        total_adv = torch.clamp(total_adv, -eps, eps)
        crafting_output = inputs + total_adv
        crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
    added = crafting_output - inputs
    added = added.view(added.size(0), 1, added.size(1))
    added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    
    for i in range(num_steps*2):
        temp = nnF.conv1d(added, weights[0], padding = sizes[0]//2)
        for j in range(len(sizes)-1):
            temp = temp + nnF.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
        temp = temp/float(len(sizes))
        
        output = model(inputs + temp.view(temp.size(0),temp.size(2)))        
        loss = criterion(output, targets)
        loss.backward()
        
        added = added + step_alpha * torch.sign(added.grad.data)
        added = torch.clamp(added, -eps, eps)
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    temp = nnF.conv1d(added, weights[0], padding = sizes[0]//2)
    for j in range(len(sizes)-1):
        temp = temp + nnF.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
    temp = temp/float(len(sizes))
    temp = temp.view(temp.size(0),temp.size(2))
    crafting_output = inputs + temp.detach()
    crafting_output_clamp = crafting_output.clone()
    """
    for i in range(crafting_output_clamp.size(0)):
        remainder = MAX_SENTENCE_LENGTH - lengths[i]
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    """
    #crafting_output_clamp = crafting_output_clamp.view(crafting_output_clamp.size(0),crafting_output_clamp.size(2))
    sys.stdout.flush()
    return  crafting_output_clamp


def test(model, device, dataloader, num_classes):
    model.eval()#set model to evaluation mode
    sample_count=0
    sample_idx_wrong=[]
    confusion=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            Z = model(X)#forward pass
            if len(Z.size()) <= 1:
                Yp = (Z>0).to(torch.int64) #binary/sigmoid
                Y = (Y>0.5).to(torch.int64)
            else:
                Yp = Z.data.max(dim=1)[1] #multiclass/softmax
            for i in range(0, num_classes):
                for j in range(0, num_classes):
                    confusion[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
            #------------------
            for n in range(0,X.size(0)):
                if Y[n] != Yp[n]:
                    sample_idx_wrong.append(sample_count+n)
            sample_count+=X.size(0)
        #------------------
    sens=np.zeros(num_classes)
    prec=np.zeros(num_classes)
    for n in range(0, num_classes):
        TP=confusion[n,n]
        FN=np.sum(confusion[n,:])-TP
        FP=np.sum(confusion[:,n])-TP
        sens[n]=TP/(TP+FN)
        prec[n]=TP/(TP+FP)
    acc = confusion.diagonal().sum()/confusion.sum()
    result={}
    result['confusion']=confusion
    result['acc']=acc
    result['sens']=sens
    result['prec']=prec
    print('testing')
    print('acc', result['acc'])
    print('sens', result['sens'])
    print('prec', result['prec'])
    return result
#%%
def test_rand(model, device, dataloader, num_classes, noise_norm):
    model.eval()#set model to evaluation mode
    sample_count=0
    adv_sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(dataloader):
            Xn = X + noise_norm*(2*torch.rand_like(X)-1)
            Xn.clamp_(0, 1)
            Xn, X, Y = Xn.to(device), X.to(device), Y.to(device)            
            Z = model(X)
            Zn = model(Xn)
            if len(Z.size()) <= 1: #binary/sigmoid
                Y = (Y>0.5).to(torch.int64)
                Yp = (Z>0).to(torch.int64) 
                Ypn = (Zn>0).to(torch.int64)
            else:  #multiclass/softmax
                Yp = Z.data.max(dim=1)[1]
                Ypn = Zn.data.max(dim=1)[1]            
            #------------------
            #do not attack x that is missclassified
            Ypn_ = Ypn.clone().detach()
            Zn_=Zn.clone().detach()
            temp=(Yp!=Y)
            Ypn_[temp]=Yp[temp]        
            Zn_[temp]=Z[temp]
            for i in range(0, confusion_noisy.shape[0]):
                for j in range(0, confusion_noisy.shape[1]):
                    confusion_noisy[i,j]+=torch.sum((Y==i)&(Ypn_==j)).item()
            #------------------
            for i in range(0, confusion_clean.shape[0]):
                for j in range(0, confusion_clean.shape[1]):
                    confusion_clean[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
            #------------------
            for m in range(0,X.size(0)):
                idx=sample_count+m            
                if Y[m] != Yp[m]:
                    sample_idx_wrong.append(idx)
                elif Ypn[m] != Yp[m]:
                    sample_idx_attack.append(idx)
            #------------------
            sample_count+=X.size(0)
            adv_sample_count+=torch.sum((Yp==Y)&(Ypn!=Y)).item()         
    #------------------
    acc_clean = confusion_clean.diagonal().sum()/confusion_clean.sum()
    acc_noisy = confusion_noisy.diagonal().sum()/confusion_noisy.sum()
    sens_clean=np.zeros(num_classes)
    prec_clean=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens_clean[m]=confusion_clean[m,m]/np.sum(confusion_clean[m,:])
        prec_clean[m]=confusion_clean[m,m]/np.sum(confusion_clean[:,m])
    sens_noisy=np.zeros(num_classes)
    prec_noisy=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens_noisy[m]=confusion_noisy[m,m]/np.sum(confusion_noisy[m,:])
        prec_noisy[m]=confusion_noisy[m,m]/np.sum(confusion_noisy[:,m])
    #------------------
    result={}
    result['method']='rand'
    result['noise_norm']=noise_norm
    result['sample_count']=sample_count
    result['adv_sample_count']=adv_sample_count
    result['sample_idx_wrong']=sample_idx_wrong
    result['sample_idx_attack']=sample_idx_attack
    result['confusion_clean']=confusion_clean
    result['acc_clean']=acc_clean
    result['sens_clean']=sens_clean
    result['prec_clean']=prec_clean
    result['confusion_noisy']=confusion_noisy
    result['acc_noisy']=acc_noisy
    result['sens_noisy']=sens_noisy
    result['prec_noisy']=prec_noisy
    #------------------
    print('testing robustness rand, adv%=', adv_sample_count/sample_count, sep='')
    print('noise_norm:', noise_norm)
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])
    return result
#%%
def ifgsm_attack_simple(model, X, Y, noise, max_iter, step):
    model.eval()#set model to evaluation mode
    Y_int64 = Y.to(torch.int64)
    Xn = X.clone().detach()
    for n in range(0, max_iter):
        Xn.requires_grad = True
        if Xn.grad is not None:
            print(Xn.grad.abs().sum().item())
        Zn = model(Xn)
        if len(Zn.size()) <= 1:
            loss = nnF.binary_cross_entropy_with_logits(Zn, Y)
        else:
            loss = nnF.cross_entropy(Zn, Y_int64)
        #loss.backward() will update dLdW
        Xn_grad=torch.autograd.grad(loss, Xn)[0]
        Xn = Xn.detach() + step*Xn_grad.sign().detach() 
        Xn = torch.clamp(Xn, 0, 1)
        Xn = Xn.detach()
    return Xn, Xn_grad
#%%
def cal_mean_L2norm_X(dataloader):
    mean_norm=0
    sample_count=0
    for batch_idx, (X, Y) in enumerate(dataloader):
        X=X.view(X.size(0), -1)
        mean_norm+=torch.sum(torch.sqrt(torch.sum(X**2, dim=1))).item()
        sample_count+=X.size(0)
    mean_norm/=sample_count
    return mean_norm   
#%%
def cal_mean_L1norm_X(dataloader):
    mean_norm=0
    sample_count=0
    for batch_idx, (X, Y) in enumerate(dataloader):
        X=X.view(X.size(0), -1)
        mean_norm+=torch.sum(torch.sum(X.abs(), dim=1)).item()
        sample_count+=X.size(0)
    mean_norm/=sample_count
    return mean_norm 
#%%
def estimate_max_iter_and_step(noise_norm, norm_type, max_noise_norm=None, X=None, dataloader=None):
    #https://arxiv.org/pdf/1607.02533v4.pdf
    if norm_type == np.inf:        
        if max_noise_norm is None:
            # X is in range of 0 ~1, so max_noise_norm is 1
            noise_level=noise_norm      
        else:
            noise_level=noise_norm/max_noise_norm  
    elif norm_type == 2:
        if max_noise_norm is not None:
            noise_level=noise_norm/max_noise_norm
        elif X is not None:
            X=X.view(X.size(0), -1)
            L2norm_mean=torch.mean(torch.sqrt(torch.sum(X**2, dim=1))).item()
            noise_level=noise_norm/L2norm_mean
        elif dataloader is not None:
            L2norm_mean=cal_mean_L2norm_X(dataloader)
            noise_level=noise_norm/L2norm_mean
        else:
            raise ValueError("not enough input")
    else:
        raise NotImplementedError("other method is not implemented.")      
    max_iter=int(min(255*noise_level+4, 1.25*255*noise_level))+1    
    step=max(1/255, noise_level/max_iter)
    return max_iter, step
#%% 
def ifgsm_attack(model, X, Y, noise_norm, norm_type=np.inf, max_iter=None, step=None,
                 scale_z=None, targeted=False, stop_if_label_change=False, stop_if_label_change_next_step=False, 
                 use_optimizer=False, output_Xn_all=False):
    if max_iter is None and step is None:
        max_iter, step = estimate_max_iter_and_step(noise_norm, norm_type, X=X)
    #set rand_init to False
    return pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step, False, None,
                      scale_z, targeted, stop_if_label_change, stop_if_label_change_next_step,
                      use_optimizer, output_Xn_all)
#%%
def clip_norm_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is noise level      
    with torch.no_grad():
        if norm_type == np.inf:
            noise.clamp_(-norm_max, norm_max)
        elif norm_type == 1: # L1norm
            N=noise.view(noise.size(0), -1)
            l1_norm= torch.sum(N.abs(), dim=1, keepdim=True)
            temp = (l1_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/l1_norm[temp]
        elif norm_type == 2: # L2 norm
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/l2_norm[temp]
        elif norm_type >= 1:
            p=norm_type
            N=noise.view(noise.size(0), -1)
            lp_norm = (torch.sum(N.abs()**p))**(1/p)
            temp = (lp_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/lp_norm[temp]
        else:
            raise NotImplementedError("other norm clip is not implemented.")  
    return noise
#%%
def normalize_grad_(x_grad, norm_type, norm_const=1, eps=1e-8):    
    #x_grad is modified in place
    #x_grad.size(0) is batch_size
    with torch.no_grad():
        if norm_type == np.inf:
            x_grad-=x_grad-norm_const*x_grad.sign()
        elif norm_type == 1: # L1norm
            g=x_grad.view(x_grad.size(0), -1)
            l1_norm=torch.sum(g.abs(), dim=1, keepdim=True)
            g *= norm_const/(l1_norm+eps)
        elif norm_type == 2: # L2 norm
            g=x_grad.view(x_grad.size(0), -1)
            l2_norm=torch.sqrt(torch.sum(g**2, dim=1, keepdim=True))
            g *= norm_const/(l2_norm+eps)
        elif norm_type >= 1:
            p=norm_type
            g=x_grad.view(x_grad.size(0), -1)
            lp_norm = (torch.sum(g.abs()**p))**(1/p)
            g *=norm_const/(lp_norm+eps)
        else:
            raise NotImplementedError("not implemented.")  
    return x_grad
#%% this is Projected Gradient Descent (PGD) attack
def pgd_attack(model, X, Y, noise_norm, norm_type=np.inf, max_iter=None, step=None, rand_init=True, rand_init_max=None, 
               scale_z=None, targeted=False, stop_if_label_change=False, stop_if_label_change_next_step=False, 
               use_optimizer=False, output_Xn_all=False):
    # X is in range of 0~1
    # noise_norm is the bound of noise, it is noise level if norm_type is inf
    #   it is L-inf norm (0~1) of max noise (Xn-X) if norm_type=np.inf
    #   it is L1 norm of max noise if norm_type=1, it is better to use SparseL1Descent (see cleverhans github)
    #   it is L2 norm of max noise if norm_type=2, it may be better to use an optimizer (set use_optimizer to True)
    # norm_type can be np.inf, 1, 2
    model.eval()#set model to evaluation mode
    X = X.detach()
    Y = Y.to(torch.int64)
    #-----------------
    if output_Xn_all == True:
        Xn_all=[]   
    #-----------------
    if stop_if_label_change == True or stop_if_label_change_next_step == True:
        with torch.no_grad():
            Z=model(X)
            if len(Z.size()) <= 1:
                Yp = (Z>0).to(torch.int64)
            else:
                Yp = Z.data.max(dim=1)[1]
            Yp_e_Y = Yp==Y
            del Z, Yp # the graph should be released
    #-----------------
    if stop_if_label_change_next_step == True:
        noise_old=torch.zeros_like(X)
        flag=torch.zeros(X.size(0), device=X.device, dtype=torch.int64)
        #flag[k]=1, X[k] will go across the decision boundary in next step
        #flag[k]=0, otherwise        
    #-----------------
    if rand_init == True:
        init_value=rand_init_max
        if rand_init_max is None:
            init_value=noise_norm            
        noise_init=init_value*(2*torch.rand_like(X)-1)
        clip_norm_(noise_init, norm_type, noise_norm)
        Xn = X + noise_init    
    else:
        Xn = X.clone().detach() # must clone
    #-----------------
    noise_new=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise_new], lr=step)
    #-----------------    
    for n in range(0, max_iter):
        Xn = Xn.detach()
        Xn.requires_grad = True
        Zn = model(Xn)
        #scale Z to avoid numerical problem (e.g. sigmoid(Z)~1, then no gradient)
        #for binary_cross_entropy, Xn.grad.sign() will not change after scale Z
        if scale_z != None:
            Zn=Zn*scale_z
        if len(Zn.size()) <= 1:
            Ypn = (Zn>0).to(torch.int64)
            loss = nnF.binary_cross_entropy_with_logits(Zn, Y.to(X.dtype))
        else:
            Ypn = Zn.data.max(dim=1)[1]
            loss = nnF.cross_entropy(Zn, Y)
        Ypn_e_Y=(Ypn==Y)
        Ypn_ne_Y=(Ypn!=Y)
        if targeted == True:#targeted attack, Y should be filled with targeted class label
            loss=-loss
        #loss.backward() will update W.grad
        grad_n=torch.autograd.grad(loss, Xn)[0]         
        grad_n=normalize_grad_(grad_n, norm_type)  
        if use_optimizer == False:
            Xnew = Xn.detach() + step*grad_n.detach()      
            noise_new = Xnew-X
        else:
            noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
            optimizer.step()
        clip_norm_(noise_new, norm_type, noise_norm)     
        Xnew = torch.clamp(X+noise_new, 0, 1)
        noise_new.data -= noise_new.data-(Xnew-X).data
        #---------------------------
        if stop_if_label_change == True:
            Xn=Xn.detach()
            if targeted == True:
                Xn[Ypn_ne_Y]=X[Ypn_ne_Y] + noise_new[Ypn_ne_Y]
            else:
                #Yp_e_Y is needed to get temp because:
                #if Yp is wrong but Ypn is correct (this is possible! I have seen such cases)
                #then Xn should not be updated
                temp=Yp_e_Y&Ypn_e_Y
                Xn[temp]=X[temp] + noise_new[temp]            
        elif stop_if_label_change_next_step == True:
            Xn=Xn.detach()
            if targeted == True:
                Xn[Ypn_ne_Y]=X[Ypn_ne_Y] + noise_new[Ypn_ne_Y]
                Xn[Ypn_e_Y]=X[Ypn_e_Y] + noise_old[Ypn_e_Y]
                noise_old[Ypn_ne_Y] = noise_new[Ypn_ne_Y]
            else:
                temp1=Yp_e_Y&Ypn_e_Y
                temp2=Yp_e_Y&Ypn_ne_Y
                Xn[temp1]=X[temp1] + noise_new[temp1]
                Xn[temp2]=X[temp2] + noise_old[temp2] # go back one step
                noise_old[temp1] = noise_new[temp1]
                flag[temp2]=1                
        else:
            Xn=X+noise_new
        #---------------------------
        Xn=torch.clamp(Xn, 0, 1)
        Xn=Xn.detach()
        if output_Xn_all == True:
            Xn_all.append(Xn.cpu())
        #---------------------------
        #print('n=', n)
    if output_Xn_all == False:
        if stop_if_label_change_next_step == False:
            return Xn
        else:
            return Xn, flag
    else:
        return Xn_all
#%%
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
    if save_model_output == True:
        y_list=[]
        z_list=[]
        yp_list=[]        
        adv_z_list=[]
        adv_yp_list=[]
    #---------------------
    if method == 'ifgsm' and max_iter is None and step is None:
        max_iter, step = estimate_max_iter_and_step(noise_norm, norm_type, dataloader=dataloader)
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
        if method == 'ifgsm':
            Xn = ifgsm_attack(model, X, Y, noise_norm, norm_type, max_iter, step, 
                              scale_z, targeted, stop_if_label_change,
                              use_optimizer=use_optimizer)
        elif method == 'pgd':
            Xn = pgd_attack(model, X, Y, noise_norm, norm_type, max_iter, step, True, None, 
                            scale_z, targeted, stop_if_label_change,
                            use_optimizer=use_optimizer)
        elif method == 'SAP':
            sizes = [5, 7, 11, 15, 19]
            sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
            #print('sizes:',sizes)
            #print('sigmas:', sigmas)
            crafting_sizes = []
            crafting_weights = []
            for size in sizes:
                for sigma in sigmas:
                    crafting_sizes.append(size)
                    weight = np.arange(size) - size//2
                    weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
                    weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
                    crafting_weights.append(weight)
            #print("crafting sizes ",crafting_sizes
            
            #print("crafting w", crafting_weights)
            Xn = pgd_conv(inputs = X, targets=Y, model=model, eps = noise_norm, step_alpha = step, 
                          num_steps = max_iter, sizes = crafting_sizes, weights = crafting_weights)
        else:
            raise NotImplementedError("other method is not implemented.")
        #------------------
        Zn = model(Xn)# classify the noisy signal Xn
        if len(Z.size()) <= 1:
            Ypn = (Zn>0).to(torch.int64)
        else:
            Ypn = Zn.data.max(dim=1)[1]
        #------------------
        if len(Z.size()) <= 1:
            Y = (Y>0.5).to(torch.int64)
        #------------------
        #do not attack x that is missclassified
        Ypn_ = Ypn.clone().detach()
        Zn_=Zn.clone().detach()
        if targeted == False:
            temp=(Yp!=Y)
            Ypn_[temp]=Yp[temp]        
            Zn_[temp]=Z[temp]
        for i in range(0, confusion_noisy.shape[0]):
            for j in range(0, confusion_noisy.shape[1]):
                confusion_noisy[i,j]+=torch.sum((Y==i)&(Ypn_==j)).item()
        #------------------
        for i in range(0, confusion_clean.shape[0]):
            for j in range(0, confusion_clean.shape[1]):
                confusion_clean[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
        #------------------
        for m in range(0,X.size(0)):
            idx=sample_count+m            
            if Y[m] != Yp[m]:
                sample_idx_wrong.append(idx)
            elif Ypn[m] != Yp[m]:
                sample_idx_attack.append(idx)
        sample_count+=X.size(0)
        adv_sample_count+=torch.sum((Yp==Y)&(Ypn!=Y)).item()
        #------------------
        if save_model_output == True:
            y_list.append(Y.detach().to('cpu').numpy())
            z_list.append(Z.detach().to('cpu').numpy())
            yp_list.append(Yp.detach().to('cpu').numpy())            
            adv_z_list.append(Zn_.detach().to('cpu').numpy())
            adv_yp_list.append(Ypn_.detach().to('cpu').numpy())
        #------------------
    #------------------
    acc_clean = confusion_clean.diagonal().sum()/confusion_clean.sum()
    acc_noisy = confusion_noisy.diagonal().sum()/confusion_noisy.sum()
    sens_clean=np.zeros(num_classes)
    prec_clean=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens_clean[m]=confusion_clean[m,m]/np.sum(confusion_clean[m,:])
        prec_clean[m]=confusion_clean[m,m]/np.sum(confusion_clean[:,m])
    sens_noisy=np.zeros(num_classes)
    prec_noisy=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens_noisy[m]=confusion_noisy[m,m]/np.sum(confusion_noisy[m,:])
        prec_noisy[m]=confusion_noisy[m,m]/np.sum(confusion_noisy[:,m])
    #------------------
    result={}
    result['method']=method
    result['noise_norm']=noise_norm
    result['norm_type']=norm_type
    result['max_iter']=max_iter
    result['step']=step
    result['sample_count']=sample_count
    result['adv_sample_count']=adv_sample_count
    result['sample_idx_wrong']=sample_idx_wrong
    result['sample_idx_attack']=sample_idx_attack
    result['confusion_clean']=confusion_clean
    result['acc_clean']=acc_clean
    result['sens_clean']=sens_clean
    result['prec_clean']=prec_clean
    result['confusion_noisy']=confusion_noisy
    result['acc_noisy']=acc_noisy
    result['sens_noisy']=sens_noisy
    result['prec_noisy']=prec_noisy
    #------------------
    if save_model_output == True:
        y_list = np.concatenate(y_list, axis=0).squeeze().astype('int64')
        z_list=np.concatenate(z_list, axis=0).squeeze()
        yp_list = np.concatenate(yp_list, axis=0).squeeze().astype('int64')
        adv_z_list=np.concatenate(adv_z_list, axis=0).squeeze()
        adv_yp_list = np.concatenate(adv_yp_list, axis=0).squeeze().astype('int64')
        result['y']=y_list
        result['z']=z_list
        result['yp']=yp_list
        result['adv_z']=adv_z_list
        result['adv_yp']=adv_yp_list
    #------------------
    print('testing robustness wba ', method, ', adv%=', adv_sample_count/sample_count, sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm, ', max_iter:', max_iter, ', step:', step, sep='')
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])
    return result
#%%