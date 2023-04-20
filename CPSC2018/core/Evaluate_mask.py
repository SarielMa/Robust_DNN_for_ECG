import numpy as np
import torch
from torch import optim
import torch.nn.functional as nnF
#%%
def cal_AUC_robustness(acc_list, noise_level_list):
    #noise_level_list[0] is 0
    #acc_list[0] is acc on clean data
    auc=0
    for n in range(1, len(acc_list)):
        auc+= (acc_list[n]+acc_list[n-1])*(noise_level_list[n]-noise_level_list[n-1])*0.5
    auc/=noise_level_list[n]
    return auc
#%%
def cal_performance(confusion, class_balanced_acc=False):
    num_classes=confusion.shape[0]
    if class_balanced_acc == True:
        confusion=confusion.copy()
        for m in range(0, num_classes):
            confusion[m]/=confusion[m].sum()+1e-8
    acc = confusion.diagonal().sum()/confusion.sum()
    sens=np.zeros(num_classes)
    prec=np.zeros(num_classes)
    for m in range(0, num_classes):
        sens[m]=confusion[m,m]/(np.sum(confusion[m,:])+1e-8)
        prec[m]=confusion[m,m]/(np.sum(confusion[:,m])+1e-8)
    return acc, sens, prec
#%%
def test(model, device, dataloader, num_classes, class_balanced_acc=False):
    model.eval()#set model to evaluation mode
    sample_count=0
    sample_idx_wrong=[]
    confusion=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            X, Y = batch_data[0].to(device), batch_data[1].to(device)
            Mask = batch_data[2].to(device)
            Z = model(X, Mask)#forward pass
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
    '''
    sens=np.zeros(num_classes)
    prec=np.zeros(num_classes)
    for n in range(0, num_classes):
        TP=confusion[n,n]
        FN=np.sum(confusion[n,:])-TP
        FP=np.sum(confusion[:,n])-TP
        sens[n]=TP/(TP+FN)
        prec[n]=TP/(TP+FP)
    acc = confusion.diagonal().sum()/confusion.sum()
    '''
    acc, sens, prec = cal_performance(confusion, class_balanced_acc)
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
def test_rand(model, device, dataloader, num_classes, noise_norm, clip_X_min=-1, clip_X_max=1, class_balanced_acc=False):
    model.eval()#set model to evaluation mode
    sample_count=0
    adv_sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            X, Y = batch_data[0].to(device), batch_data[1].to(device)
            Mask = batch_data[2].to(device)
            Xn = X + noise_norm*(2*torch.rand_like(X)-1)
            Xn.clamp_(clip_X_min, clip_X_max)
            Z = model(X, Mask)
            Zn = model(Xn, Mask)
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
    acc_clean, sens_clean, prec_clean = cal_performance(confusion_clean, class_balanced_acc)
    acc_noisy, sens_noisy, prec_noisy = cal_performance(confusion_noisy, class_balanced_acc)
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
def logit_margin_loss_binary(Z, Y, reduction='sum'):
    t = Y*2-1
    loss =- Z*t
    if reduction == 'sum':
        loss=loss.sum()
    elif reduction == 'mean':
        loss=loss.mean()
    else:
        raise NotImplementedError("other reduction is not implemented.")
    return loss
#%%
def logit_margin_loss(Z, Y, reduction='sum'):
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y[:,None])
    Zy=Zy.view(-1)
    #--------------------------------------------------------
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))
    #--------------------------------------------------------
    Zother_max = Zother.max(dim=1)[0]
    loss=Zother_max-Zy
    if reduction == 'mean':
        loss=loss.mean()
    elif reduction == 'sum':
        loss=loss.sum()
    elif reduction == 'sum_squared':
        loss=torch.sum(loss**2)
    elif reduction == 'sum_abs':
        loss=torch.sum(loss.abs())
    elif reduction == 'mean_squared':
        loss=torch.mean(loss**2)
    elif reduction == None:
        pass
    else:
        raise NotImplementedError("unkown reduction")
    return loss
#%%
def soft_logit_margin_loss(Z, Y, reduction='sum'):
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y[:,None])
    Zy=Zy.view(-1)
    #--------------------------------------------------------
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))
    #--------------------------------------------------------
    loss=torch.logsumexp(Zother, dim=1)-Zy
    #print(loss.shape)
    if reduction == 'mean':
        loss=loss.mean()
    elif reduction == 'sum':
        loss=loss.sum()
    elif reduction == 'sum_squared':
        loss=torch.sum(loss**2)
    elif reduction == 'sum_abs':
        loss=torch.sum(loss.abs())
    elif reduction == 'mean_squared':
        loss=torch.mean(loss**2)
    elif reduction == None:
        pass
    else:
        raise NotImplementedError("unkown reduction")
    return loss
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
def clip_norm_(noise, norm_type, norm_max):
    if not isinstance(norm_max, torch.Tensor):
        clip_normA_(noise, norm_type, norm_max)
    else:
        clip_normB_(noise, norm_type, norm_max)
#%%
def clip_normA_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is noise level
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf:
            noise.clamp_(-norm_max, norm_max)
        elif norm_type == 2: # L2 norm
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("other norm clip is not implemented.")
    #-----------
    return noise
#%%
def clip_normB_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max[k] is noise level for every noise[k]
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf:
            #for k in range(noise.size(0)):
            #    noise[k].clamp_(-norm_max[k], norm_max[k])
            N=noise.view(noise.size(0), -1)
            norm_max=norm_max.view(norm_max.size(0), -1)
            N=torch.max(torch.min(N, norm_max), -norm_max)
            N=N.view(noise.size())
            noise-=noise-N
        elif norm_type == 2: # L2 norm
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            norm_max=norm_max.view(norm_max.size(0), 1)
            #print(l2_norm.shape, norm_max.shape)
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                norm_max=norm_max[temp]
                norm_max=norm_max.view(norm_max.size(0), -1)
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("not implemented.")
        #-----------
    return noise
#%%
def add_noise_to_X_and_clip_norm_(noise, norm_type, norm_max, X, clip_X_min=0, clip_X_max=1):
    #noise and X are modified in place
    if X.size(0) == 0:
        return noise, X
    with torch.no_grad():
        clip_norm_(noise, norm_type, norm_max)
        Xnew = torch.clamp(X+noise, clip_X_min, clip_X_max)
        noise -= noise-(Xnew-X)
        X -= X-Xnew
    return noise, X
#%%
def normalize_grad_(x_grad, norm_type, eps=1e-8):
    #x_grad is modified in place
    #x_grad.size(0) is batch_size
    with torch.no_grad():
        if norm_type == np.inf:
            x_grad-=x_grad-x_grad.sign()
        elif norm_type == 2: # L2 norm
            g=x_grad.view(x_grad.size(0), -1)
            l2_norm=torch.sqrt(torch.sum(g**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            g *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return x_grad
#%%
def normalize_noise_(noise, norm_type, eps=1e-8):
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        N=noise.view(noise.size(0), -1)
        if norm_type == np.inf:
            linf_norm=N.abs().max(dim=1, keepdim=True)[0]
            N *= 1/(linf_norm+eps)
        elif norm_type == 2:
            l2_norm=torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
            N *= 1/l2_norm
        else:
            raise NotImplementedError("not implemented.")
    return noise
#%%
def get_noise_init(norm_type, noise_norm, init_value, X):
    noise_init=2*torch.rand_like(X)-1
    noise_init=noise_init.view(X.size(0),-1)
    if isinstance(init_value, torch.Tensor):
        init_value=init_value.view(X.size(0), -1)
    noise_init=init_value*noise_init
    noise_init=noise_init.view(X.size())
    clip_norm_(noise_init, norm_type, init_value)
    clip_norm_(noise_init, norm_type, noise_norm)
    return noise_init
#%%
def ifgsm_attack_original(model, X, Y, Mask, noise, max_iter=None, step=None,
                          model_eval_attack=False, return_Xn_grad=False):
    if max_iter is None and step is None:
        max_iter=int(min(255*noise+4, 1.25*255*noise))
        step=1/255
    #-----------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True:
        model.eval()#set model to evaluation mode
    Y_int64 = Y.to(torch.int64)
    Xn = X.clone().detach()
    for n in range(0, max_iter):
        Xn.requires_grad = True
        if Xn.grad is not None:
            print(Xn.grad.abs().sum().item())
        Zn = model(Xn, Mask)
        if len(Zn.size()) <= 1:
            loss = nnF.binary_cross_entropy_with_logits(Zn, Y)
        else:
            loss = nnF.cross_entropy(Zn, Y_int64)
        #loss.backward() will update dLdW
        Xn_grad=torch.autograd.grad(loss, Xn)[0]
        Xn = Xn.detach() + step*Xn_grad.sign().detach()
        Xn = torch.clamp(Xn, 0, 1)
        Xn = Xn.detach()
    if model_eval_attack == True and train_mode == True:
        model.train()
    if return_Xn_grad == False:
        return Xn
    else:
        return Xn, Xn_grad
#%%
def get_pgd_loss_fn_by_name(loss_fn):
    if loss_fn is None:
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
    elif isinstance(loss_fn, str):
        if loss_fn == 'ce':
            loss_fn=torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_fn == 'bce':
            loss_fn=torch.nn.BCEWithLogitsLoss(reduction="sum")
        elif loss_fn =='logit_margin_loss_binary' or loss_fn == 'lmb':
            loss_fn=logit_margin_loss_binary
        elif loss_fn == 'logit_margin_loss' or loss_fn == 'lm':
            loss_fn=logit_margin_loss
        elif loss_fn == 'soft_logit_margin_loss' or loss_fn == 'slm':
            loss_fn=soft_logit_margin_loss
        else:
            raise NotImplementedError("not implemented.")
    return loss_fn
#%%
def pgd_attack_original(model, X, Y, Mask, noise_norm, norm_type, max_iter, step,
                        rand_init=True, rand_init_norm=None, targeted=False,
                        clip_X_min=0, clip_X_max=1, use_optimizer=False,
                        loss_fn=None, model_eval_attack=False):
    #-----------------------------------------------------
    loss_fn=get_pgd_loss_fn_by_name(loss_fn)
    #-----------------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#set model to evaluation mode
    #-----------------
    X = X.detach()
    #-----------------
    if rand_init == True:
        init_value=rand_init_norm
        if rand_init_norm is None:
            init_value=noise_norm
        noise_init=get_noise_init(norm_type, noise_norm, init_value, X)
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
        Zn = model(Xn, Mask)
        loss = loss_fn(Zn, Y)
        #---------------------------
        if targeted == True:
            loss=-loss
        #---------------------------
        #loss.backward() will update W.grad
        grad_n=torch.autograd.grad(loss, Xn)[0]
        grad_n=normalize_grad_(grad_n, norm_type)
        if use_optimizer == True:
            noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
            optimizer.step()
        else:
            Xnew = Xn.detach() + step*grad_n.detach()
            noise_new = Xnew-X
        #---------------------
        noise_new.data*=Mask
        #---------------------
        clip_norm_(noise_new, norm_type, noise_norm)
        Xn = torch.clamp(X+noise_new, clip_X_min, clip_X_max)
        noise_new.data -= noise_new.data-(Xn-X).data
        Xn=Xn.detach()
    #---------------------------
    if train_mode == True and model.training == False:
        model.train()
    #---------------------------
    return Xn
#%%
def ifgsm_attack(model, X, Y, Mask, noise_norm, norm_type=np.inf, max_iter=None, step=None,
                 targeted=False, clip_X_min=0, clip_X_max=1,
                 stop_if_label_change=False, stop_if_label_change_next_step=False,
                 use_optimizer=False, loss_fn=None, model_eval_attack=False,
                 return_output=False, return_Xn_all=False):
    #https://arxiv.org/pdf/1607.02533v4.pdf
    if max_iter is None and step is None:
        max_iter=int(min(255*noise_norm+4, 1.25*255*noise_norm))
        step=1/255
    #set rand_init to False
    return pgd_attack(model, X, Y, Mask, noise_norm, norm_type, max_iter, step,
                      rand_init=False, rand_init_norm=None, rand_init_Xn=None,
                      targeted=targeted, clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                      stop_if_label_change=stop_if_label_change,
                      stop_if_label_change_next_step=stop_if_label_change_next_step,
                      use_optimizer=use_optimizer, loss_fn=loss_fn, model_eval_attack=model_eval_attack,
                      return_output=return_output, return_Xn_all=return_Xn_all)
#%% this is Projected Gradient Descent (PGD) attack
def pgd_attack(model, X, Y, Mask, noise_norm, norm_type=np.inf, max_iter=None, step=None,
               rand_init=True, rand_init_norm=None, rand_init_Xn=None,
               targeted=False, clip_X_min=0, clip_X_max=1,
               stop_if_label_change=False, stop_if_label_change_next_step=False, no_attack_on_wrong=False,
               use_optimizer=False, loss_fn=None, model_eval_attack=False,
               return_output=False, return_Xn_all=False):
    # X is in range of clip_X_min ~ clip_X_max
    # noise_norm is the bound of noise
    # norm_type can be np.inf, 2
    # attack for nomr_type 1 is not implemented
    # set rand_init to False, it becomes ifgsm_attack
    #-----------------------------------------------------
    loss_fn=get_pgd_loss_fn_by_name(loss_fn)
    #-----------------------------------------------------
    train_mode=model.training# record the mode
    if model_eval_attack == True and train_mode == True:
        model.eval()#set model to evaluation mode
    #-----------------
    X = X.detach()
    Y = Y.to(torch.int64)
    #-----------------
    if return_Xn_all == True:
        Xn_all=[]
    #-----------------
    if stop_if_label_change or stop_if_label_change_next_step or no_attack_on_wrong:
        with torch.no_grad():
            Z=model(X, Mask)
            if len(Z.size()) <= 1:
                Yp = (Z>0).to(torch.int64)
            else:
                Yp = Z.data.max(dim=1)[1]
            Yp_e_Y = Yp==Y
            del Z, Yp # the graph should be released
    else:
        Yp_e_Y=Y==Y
    #-----------------
    noise_old=torch.zeros_like(X)
    flag=torch.zeros(X.size(0), device=X.device, dtype=torch.int64)
    #flag[k]=1, X[k] will go across the decision boundary in next step
    #flag[k]=0, otherwise
    #-----------------
    if rand_init == True:
        init_value=rand_init_norm
        if rand_init_norm is None:
            init_value=noise_norm
        if rand_init_Xn is None:
            noise_init=get_noise_init(norm_type, noise_norm, init_value, X)
        else:
            noise_init=(rand_init_Xn-X).detach()
        Xn = X + noise_init
    else:
        Xn = X.clone().detach() # must clone
    #-----------------
    noise_new=(Xn-X).detach()
    if use_optimizer == True:
        optimizer = optim.Adamax([noise_new], lr=step)
    #-----------------
    for n in range(0, max_iter+1):
        Xn = Xn.detach()
        Xn.requires_grad = True
        Zn = model(Xn, Mask)
        if len(Zn.size()) <= 1:
            Ypn = (Zn>0).to(torch.int64)
            loss = loss_fn(Zn, Y.to(X.dtype))
        else:
            Ypn = Zn.data.max(dim=1)[1]
            loss = loss_fn(Zn, Y)
        Ypn_e_Y=(Ypn==Y)
        Ypn_ne_Y=(Ypn!=Y)
        #---------------------------
        #targeted attack, Y should be filled with targeted class label
        if targeted == False:
            #Yp_e_Y is needed to get temp because:
            #if Yp is wrong but Ypn is correct (this is possible! I have seen such cases)
            #then Xn should not be updated
            A=Yp_e_Y&Ypn_e_Y
            B=Yp_e_Y&Ypn_ne_Y
        else:
            A=Ypn_ne_Y
            B=Ypn_e_Y
            loss=-loss
        #---------------------------
        noise_old[A] = noise_new[A].data
        #---------------------------
        if n < max_iter:
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=normalize_grad_(grad_n, norm_type)
            if use_optimizer == True:
                noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
                optimizer.step()
            else:
                Xnew = Xn.detach() + step*grad_n.detach()
                noise_new = Xnew-X
            #---------------------
            noise_new.data*=Mask
            #---------------------
            clip_norm_(noise_new, norm_type, noise_norm)
        #---------------------------
        Xn=Xn.detach()
        if stop_if_label_change == True:
            Xn[A]=X[A]+noise_new[A]
        elif stop_if_label_change_next_step == True:
            Xn[A]=X[A]+noise_new[A]
            Xn[B]=X[B]+noise_old[B] # go back
            flag[B]=1
        else:
            Xn = X+noise_new
        #---------------------------
        Xn = torch.clamp(Xn, clip_X_min, clip_X_max)
        noise_new.data -= noise_new.data-(Xn-X).data
        #---------------------------
        Xn=Xn.detach()
        if return_Xn_all == True:
            Xn_all.append(Xn.cpu())
    #---------------------------
    if train_mode == True and model.training == False:
        model.train()
    #---------------------------
    if return_output == True:
        if stop_if_label_change_next_step == False:
            Xn=(Xn, Zn.detach(), Ypn.detach())
        else:
            raise NotImplementedError("not implemented.")
    #---------------------------
    if return_Xn_all == False:
        if stop_if_label_change_next_step == False:
            return Xn
        else:
            return Xn, flag
    else:
        return Xn_all
#%%
def repeated_pgd_attack(model, X, Y, Mask, noise_norm, norm_type, max_iter, step,
                        rand_init_norm=None, targeted=False, clip_X_min=0, clip_X_max=1,
                        stop_if_label_change=True, use_optimizer=False, loss_fn=None, model_eval_attack=False,
                        return_output=False, num_repeats=0):
    for m in range(0, num_repeats+1):
        Xm, Zm, Ypm = pgd_attack(model, X, Y, Mask, noise_norm, norm_type=norm_type, max_iter=max_iter, step=step,
                                 rand_init=True, rand_init_norm=rand_init_norm,
                                 targeted=targeted, clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                 stop_if_label_change=stop_if_label_change, use_optimizer=use_optimizer,
                                 loss_fn=loss_fn, model_eval_attack=model_eval_attack, return_output=True)
        if m == 0:
            Xn=Xm
            Zn=Zm
            Ypn=Ypm
        else:
            if targeted==False:
                adv=Ypm!=Y
            else:
                adv=Ypm==Y
            Xn[adv]=Xm[adv].data
            Zn[adv]=Zm[adv].data
            Ypn[adv]=Ypm[adv].data
    #--------
    if return_output == False:
        return Xn
    else:
        return Xn, Zn, Ypn
#%%
def test_adv(model, device, dataloader, num_classes, noise_norm, norm_type, max_iter, step, method,
             targeted=False, clip_X_min=0, clip_X_max=1,
             stop_if_label_change=True, use_optimizer=False, pgd_loss_fn=None, num_repeats=0,
             save_model_output=False, class_balanced_acc=False):
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
    print('testing robustness wba ', method, '(', num_repeats, ')', sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm, ', max_iter:', max_iter, ', step:', step, sep='')
    pgd_loss_fn=get_pgd_loss_fn_by_name(pgd_loss_fn)
    print('pgd_loss_fn', pgd_loss_fn)
    #---------------------
    for batch_idx, batch_data in enumerate(dataloader):
        X, Y = batch_data[0].to(device), batch_data[1].to(device)
        Mask = batch_data[2].to(device)
        #------------------
        Z = model(X, Mask)#classify the 'clean' signal X
        if len(Z.size()) <= 1:
            Yp = (Z>0).to(torch.int64) #binary/sigmoid
        else:
            Yp = Z.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        if method == 'ifgsm':
            Xn, Zn, Ypn = ifgsm_attack(model, X, Y, Mask, noise_norm=noise_norm, norm_type=norm_type,
                                       max_iter=max_iter, step=step, targeted=targeted,
                                       clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                       stop_if_label_change=stop_if_label_change,
                                       use_optimizer=use_optimizer, loss_fn=pgd_loss_fn, return_output=True)
        elif method == 'pgd':
            Xn, Zn, Ypn = repeated_pgd_attack(model, X, Y, Mask, noise_norm=noise_norm, norm_type=norm_type,
                                              max_iter=max_iter, step=step, targeted=targeted,
                                              clip_X_min=clip_X_min, clip_X_max=clip_X_max,
                                              stop_if_label_change=stop_if_label_change,
                                              use_optimizer=use_optimizer, loss_fn=pgd_loss_fn,
                                              return_output=True, num_repeats=num_repeats)
        else:
            raise NotImplementedError("other method is not implemented.")
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
    acc_clean, sens_clean, prec_clean = cal_performance(confusion_clean, class_balanced_acc)
    acc_noisy, sens_noisy, prec_noisy = cal_performance(confusion_noisy, class_balanced_acc)
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
    print('testing robustness wba ', method, '(', num_repeats, '), adv%=', adv_sample_count/sample_count, sep='')
    print('norm_type:', norm_type, ', noise_norm:', noise_norm, ', max_iter:', max_iter, ', step:', step, sep='')
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])
    print ('noisy confusion matrix is ',confusion_noisy)
    return result
#%%