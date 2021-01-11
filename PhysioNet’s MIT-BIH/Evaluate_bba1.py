# adversarialbox.attacks part=================================================================================
#PyTorch Implementation of Papernot's Black-Box Attack
#arXiv:1602.02697

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
#%%
def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end
#%%
def label_data(oracle, device, X_sub, batch_size):
    nb_batches = int(np.ceil(float(X_sub.shape[0]) / batch_size))
    Y_sub=np.zeros(X_sub.shape[0], dtype=np.int64)
    for batch in range(nb_batches):
        start, end = batch_indices(batch, len(X_sub), batch_size)
        X = X_sub[start:end]
        X = torch.from_numpy(X).to(device)
        Z = oracle(X)
        if len(Z.size()) <= 1:
            Y = (Z>0).to(torch.int64) #binary/sigmoid
        else:
            Y = Z.data.max(dim=1)[1] #multiclass/softmax
        Y_sub[start:end]=Y.data.cpu().numpy()
    return Y_sub
#%%
def jacobian_augmentation(sub_model, device, X_sub, Y_sub, nb_classes, batch_size, lmbda=0.1):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    sub_model.eval()
    X_aug = np.zeros(X_sub.shape, dtype=np.float32)
    nb_batches = int(np.ceil(float(len(X_sub))/batch_size))
    for batch in range(nb_batches):
        start, end = batch_indices(batch, len(X_sub), batch_size)
        X = X_sub[start:end]; X = torch.from_numpy(X).to(device)
        Yc = Y_sub[start:end]; Yg = torch.from_numpy(Yc).to(device)
        Xa= X_aug[start:end]
        for n in range(nb_classes):
            Xn=X[Yg==n].detach()
            if Xn.size(0) > 0:
                Xn.requires_grad=True
                Zn=sub_model(Xn)
                #Zn=nnF.softmax(Zn, dim=1), not necessary
                Zn=Zn[:,n].sum()
                #Zn.backward() will update dLdW
                Xn_grad=torch.autograd.grad(Zn, Xn)[0]                     
                Xn_aug=Xn.data+lmbda*Xn_grad.sign().detach()
                Xn_aug.clamp_(0, 1)
                Xa[Yc==n]=Xn_aug.data.cpu().numpy()
    #--------------------------------------------------------
    return X_aug
#%%
def train_sub_model(sub_model, oracle, device, dataloader, param, print_msg=False):
    #----------------------------------------------------
    data_iter = iter(dataloader)
    X_sub=[]
    initial_size=0
    while True:
        X_sub.append(data_iter.next()[0])
        initial_size+=X_sub[-1].size(0)
        if print_msg == True:
            print(initial_size)
        if  initial_size >= param['dataset_initial_size']:
            break
    X_sub=torch.cat(X_sub, dim=0)
    X_sub=X_sub.data.cpu().numpy().astype(np.float32)
    #----------------------------------------------------
    sub_model.to(device)    
    sub_model.train()
    oracle.to(device)
    oracle.eval()
    #----------------------------------------------------
    # Setup training
    optimizer = optim.Adamax(sub_model.parameters(), lr=param['learning_rate'], weight_decay=0)
    # initial training data X_sub
    rng = np.random.RandomState(0)
    # Train the substitute SModel and augment dataset alternatively
    for rho in range(param['num_data_aug']):
        #Label the substitute training set
        Y_sub=label_data(oracle, device, X_sub, param['batch_size'])
        # train sub_model on X_sub, Y_sub
        sub_model.train()
        for epoch in range(param['num_epochs']):
            nb_batches = int(np.ceil(float(X_sub.shape[0]) / param['batch_size']))
            assert nb_batches * param['batch_size'] >= X_sub.shape[0]
            # Indices to shuffle training set
            index_shuf = list(range(X_sub.shape[0]))
            rng.shuffle(index_shuf)
            loss_epoch=0
            acc_epoch=0
            for batch in range(nb_batches):
                start, end = batch_indices(batch, X_sub.shape[0], param['batch_size'])
                X = X_sub[index_shuf[start:end]]; X = torch.from_numpy(X).to(device)
                Y = Y_sub[index_shuf[start:end]]; Y = torch.from_numpy(Y).to(device)                
                Zs = sub_model(X)
                Yp = Zs.data.max(dim=1)[1] #multiclass/softmax
                loss = nnF.cross_entropy(Zs, Y)
                optimizer.zero_grad()
                loss.backward()
                weight_decay(optimizer,param['weight_decay'])
                optimizer.step()
                loss_epoch+=loss.item()
                acc_epoch+= torch.sum(Yp==Y).item()
            loss_epoch/=nb_batches
            acc_epoch/=X_sub.shape[0]
            if print_msg == True:
                print('Zs.abs().max()', Zs.abs().max().item())
                print('Train_Substitute: aug', rho, 'epoch', epoch, 'loss', loss_epoch, 'acc', acc_epoch)
        if rho == param['num_data_aug']-1:
            break
        #Perform Jacobian-based dataset augmentation
        sub_model.eval()
        if X_sub.shape[0] < param['dataset_max_size']:
            lmbda_new = 2 * int(int(rho / 3) != 0) - 1
            lmbda_new *= 0.1 
            X_aug = jacobian_augmentation(sub_model, device, X_sub, Y_sub, param['num_classes'], param['batch_size'], lmbda=lmbda_new)
            X_sub = np.concatenate([X_sub, X_aug], axis=0)
        else:
            #augment half of the data
            X_sub_a = X_sub[index_shuf[0:int(X_sub.shape[0]/2)]]
            Y_sub_a = Y_sub[index_shuf[0:int(X_sub.shape[0]/2)]]
            X_aug = jacobian_augmentation(sub_model, device, X_sub_a, Y_sub_a, param['num_classes'], param['batch_size'])
            X_sub_b = X_sub[index_shuf[int(X_sub.shape[0]/2):]]
            #combine old and aug data
            X_sub = np.concatenate([X_sub_b, X_aug], axis=0)
        if print_msg == True:
            print(X_sub.shape)
#%%
def weight_decay(optimizer, rate):
    with torch.no_grad():
        for g in optimizer.param_groups:
            lr=g['lr']
            for p in g['params']:
                if p.requires_grad == True:
                    p -= lr*rate*p
#%%
def ifgsm_attack(sub_model, oracle, X, Y, noise_norm, norm_type=np.inf, max_iter=None, step=None, 
                 scale_z=None, targeted=False, stop_if_label_change=False):
    #https://arxiv.org/pdf/1607.02533v4.pdf
    if norm_type == np.inf:
        if max_iter is None:
            max_iter=int(min(255*noise_norm+4, 1.25*255*noise_norm))
        if step is None:
            step=max(1/255, noise_norm/max_iter)
    #set rand_init to False
    return pgd_attack(sub_model, oracle, X, Y, noise_norm, norm_type, max_iter, step, 
                      False, None, scale_z, targeted, stop_if_label_change)
#%%
def clip_norm_(noise, norm_type, norm_max):
    # noise is a tensor modified in place
    # norm_type can be np.inf, 1 or 2
    # norm_max is noise level      
    n_size=noise.size()
    if norm_type == np.inf:
        noise.clamp_(-norm_max, norm_max)
    elif norm_type == 1: # L1norm
        noise=noise.view(n_size[0], -1)
        l1_norm= torch.sum(noise.abs(), dim=1, keepdim=True)
        temp = l1_norm > norm_max
        if temp.sum() > 0:
            noise[temp]*=norm_max/l1_norm[temp]
        noise=noise.view(n_size)
    elif norm_type == 2: # L2 norm
        noise=noise.view(n_size[0], -1)
        l2_norm= torch.sqrt(torch.sum(noise**2, dim=1, keepdim=True))
        temp = l2_norm > norm_max
        if temp.sum() > 0:
            noise[temp]*=norm_max/l2_norm[temp]
        noise=noise.view(n_size)
    else:
        raise NotImplementedError("other norm clip is not implemented.")  
    return noise
#%% this is Projected Gradient Descent (PGD) attack
def pgd_attack(sub_model, oracle, X, Y, noise_norm, norm_type=np.inf, max_iter=None, step=None,
               rand_init=True, rand_init_max=None, scale_z=None, targeted=False, stop_if_label_change=False):
    # X is in range of 0~1
    # noise is noise level
    #   it is L-inf norm (0~1) of max noise (Xn-X) if norm=np.inf
    #   it is L1 norm of max noise if norm=1 
    #   it is L2 norm of max noise if norm=2 
    # norm can be np.inf, 1, 2
    sub_model.eval()#set  to evaluation mode
    oracle.eval()#set to evaluation mode
    X = X.detach()
    Xn = X.clone().detach() # must clone
    Y_int64 = Y.to(torch.int64)
    #-----------------
    #https://arxiv.org/pdf/1607.02533v4.pdf
    if norm_type == np.inf:
        if max_iter is None:
            max_iter=int(min(255*noise_norm+4, 1.25*255*noise_norm))
        if step is None:
            step=max(1/255, noise_norm/max_iter)
    #-----------------
    if stop_if_label_change == True:
        Z=oracle(Xn)
        Z=Z.detach()# cut graph
        if len(Z.size()) <= 1:
            Yp = (Z>0).to(torch.int64)
        else:
            Yp = Z.data.max(dim=1)[1]
        Yp_e_Y = (Yp==Y_int64)
    #-----------------
    if rand_init == True:
        if rand_init_max is None:
            noise_init=noise_norm*(2*torch.rand_like(X)-1)
        else:
            noise_init=rand_init_max*(2*torch.rand_like(X)-1)
        noise_init=clip_norm_(noise_init, norm_type, noise_norm)
        Xn = Xn + noise_init
    #-----------------
    for n in range(0, max_iter):
        Xn = Xn.detach()
        Xn.requires_grad = True
        sub_model.zero_grad()
        Zn = sub_model(Xn)
        #scale Z to avoid numerical problem (e.g. sigmoid(Z)~1, then no gradient)
        #for binary_cross_entropy, Xn.grad.sign() will not change after scale Z
        if scale_z is not None:
            Zn=Zn*scale_z
        if len(Zn.size()) <= 1:
            #Ypn = (Z>0).to(torch.int64)
            loss = nnF.binary_cross_entropy_with_logits(Zn, Y)
        else:
            #Ypn = Zn.data.max(dim=1)[1]
            loss = nnF.cross_entropy(Zn, Y_int64)
        if targeted == True:#targeted attack, Y should be filled with targeted class label
            loss=-loss
        #loss.backward() will update dLdW
        Xn_grad=torch.autograd.grad(loss, Xn)[0]                     
        Xnew = Xn.detach() + step*Xn_grad.sign().detach()        
        noise_new = Xnew-X
        #clip noise_new in a norm ball
        noise_new = clip_norm_(noise_new, norm_type, noise_norm)
        #-------------------------------
        Zo = oracle(Xn.detach())
        if len(Zo.size()) <= 1:
            Ypo = (Zo>0).to(torch.int64)
        else:
            Ypo = Zo.data.max(dim=1)[1]
        #---------------------------
        if stop_if_label_change == True:
            if targeted == True:
                candidate=(Ypo!=Y_int64)
            else:
                candidate=(Yp_e_Y)&(Ypo==Y_int64)
            Xn=Xn.detach()
            Xn[candidate]=X[candidate] + noise_new[candidate]
        else:
            Xn=X+noise_new
        #---------------------------
        Xn=torch.clamp(Xn, 0, 1)
        Xn=Xn.detach()
        #print('n=', n)
    return Xn
#%%
def test_adv(sub_model, oracle, device, dataloader, num_classes, 
             noise_norm, max_iter=None, step=None, norm_type=np.inf, method='ifgsm', 
             scale_z=None, targeted=False, stop_if_label_change=True):
    sub_model.eval()#set model to evaluation mode
    oracle.eval()#set model to evaluation mode
    #----------------------------------------------------
    confusion_clean=np.zeros((num_classes,num_classes))
    confusion_noisy=np.zeros((num_classes,num_classes))
    sample_count=0
    sample_idx_wrong=[]
    sample_idx_attack=[]
    #---------------------
    #https://arxiv.org/pdf/1607.02533v4.pdf
    if norm_type == np.inf and method == 'ifgsm':
        if max_iter is None:
            max_iter=int(min(255*noise_norm+4, 1.25*255*noise_norm))
        if step is None:
            step=max(1/255, noise_norm/max_iter)
    for batch_idx, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        #------------------
        Z = oracle(X)
        if method == 'ifgsm':
            Xn = ifgsm_attack(sub_model, oracle, X, Y, noise_norm, norm_type, max_iter, step, 
                              scale_z, targeted, stop_if_label_change)
        elif method == 'pgd':
            Xn = pgd_attack(sub_model, oracle, X, Y, noise_norm, norm_type, max_iter, step, True, None,
                            scale_z, targeted, stop_if_label_change)
        else:
            raise NotImplementedError("other method is not implemented.")
        Zn = oracle(Xn)
        #------------------
        if len(Z.size()) <= 1:
            Y = (Y>0.5).to(torch.int64)
            Yp = (Z>0).to(torch.int64)
            Ypn = (Zn>0).to(torch.int64)
        else:
            Y = Y.to(torch.int64)
            Yp = Z.data.max(dim=1)[1] #multiclass/softmax
            Ypn = Zn.data.max(dim=1)[1] #multiclass/softmax
        #------------------
        #do not attack x that is missclassified
        Ypn_ = Ypn.clone().detach()
        if targeted == False:
            temp=(Yp!=Y)
            Ypn_[temp]=Yp[temp]
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
    result={}
    result['method']=method
    result['noise_norm']=noise_norm
    result['max_iter']=max_iter
    result['step']=step
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
    print('testing robustness')
    print('noise_norm', noise_norm, 'max_iter', max_iter, 'step', step)
    print('acc_clean', result['acc_clean'], ', acc_noisy', result['acc_noisy'])
    print('sens_clean', result['sens_clean'])
    print('sens_noisy', result['sens_noisy'])
    print('prec_clean', result['prec_clean'])
    print('prec_noisy', result['prec_noisy'])
    return result