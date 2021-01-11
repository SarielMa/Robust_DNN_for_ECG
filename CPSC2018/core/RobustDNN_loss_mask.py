import torch
import torch.nn.functional as nnF
import warnings
import numpy as np
#%%
def HingeLoss(Z, Y, margin, within_margin=False, use_log=False, reduction='mean'):
    #Z=model(X), Z is 1D
    half_margin=margin/2
    t = Y*2-1
    Zt = half_margin - Z*t
    if within_margin == False:
        Zt=Zt[Zt>0]
    else:
        Zt=Zt[(Zt>0)&(Zt<margin)]
    #note Zt could be empty, so torch.mean(Zt) is nan
    counter=0
    if Zt.size(0)==0:
        loss = torch.tensor(0, dtype=Z.dtype, device=Z.device, requires_grad=True)
    else:
        if use_log == False:
            loss = torch.sum(Zt)
        else:
            loss = torch.sum(torch.log(1+Zt))
        counter=Zt.size(0)
    #----------------------------
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/Z.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def multi_margin_loss(Z, Y, margin, within_margin, use_log=False, reduction='mean'):    
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y.view(-1,1))
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist=torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))
    loss = margin-Zy+Zother
    mask = torch.zeros_like(loss)
    if within_margin == False:
        mask[loss>0]=1
    else:
        mask[(loss>0)&(loss<2*margin)]=1
    #---------------------
    loss=loss*mask
    counter=mask.sum()
    #---------------------
    if use_log == True:
         loss=torch.log(1+loss)
    #---------------------            
    if reduction == 'mean':
        loss=loss.mean()
    elif reduction == 'counter':
        loss=loss.sum()
        if counter>1:
            loss=loss/counter
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def Z_loss0(Z, Y):
    loss=torch.tensor(0.0, dtype=Z.dtype, device=Z.device, requires_grad=True)
    num_classes=Z.size(1)
    for n in range(0, num_classes):
        Zn=Z[Y==n]
        Zother=torch.cat([Zn[:,0:n], Zn[:,n+1:]], dim=1)
        if Zother.size(0) > 0:
            loss=loss+torch.sum(Zother**2)
    loss=loss/(Z.size(0)*(num_classes-1))
    return loss
#%%
def Z_loss1(Z, Y, num_classes, alpha=1):
    loss=torch.tensor(0.0, dtype=Z.dtype, device=Z.device, requires_grad=True)
    num_classes=Z.size(1)
    for n in range(0, num_classes):
        Zn=Z[Y==n]
        Zn=Zn[:,n]
        if Zn.size(0) > 0:
            loss=loss+torch.sum((Zn-alpha)**2)
    loss=loss/Zn.size(0)
    return loss
#%%
def Z_loss_sphere(Z, Y, center, radius, use_log=False, reduction='mean'):
    if type(center) is not torch.Tensor:
        center = torch.tensor(center, dtype=Z.dtype, device=Z.device)
    else:
        center = center.to(Z.device)
    if type(radius) is not torch.Tensor:
        radius = torch.tensor(radius, dtype=Z.dtype, device=Z.device)
    else:
        radius = radius.to(Z.device)
    radius_sq = radius**2
    if len(radius_sq.size()) > 0:
        radius_sq = radius_sq[Y]
    #-----------------------------
    M = center[Y]
    if len(Z.size()) <= 1:
        dist_sq = (Z-M)**2
    else:
        dist_sq = torch.sum((Z-M)**2, dim=1)
    dist_sq = dist_sq[dist_sq>radius_sq]
    counter=0
    if dist_sq.size(0) == 0:
        loss=torch.tensor(0.0, dtype=Z.dtype, device=Z.device, requires_grad=True)
    else:
        if use_log == False:
            loss = torch.sum(dist_sq)
        else:
            loss = torch.sum(torch.log(1+dist_sq))
        counter=dist_sq.size(0)
    #-------------------------
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/Z.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def Z_loss_cube(Z, Y, center, radius, use_log=False, reduction='mean'):
    if type(center) is not torch.Tensor:
        center = torch.tensor(center, dtype=Z.dtype, device=Z.device)
    else:
        center = center.to(Z.device)
    if type(radius) is not torch.Tensor:
        radius = torch.tensor(radius, dtype=Z.dtype, device=Z.device)
    else:
        radius = radius.to(Z.device)
    if len(radius.size()) > 0:
        radius = radius[Y]
    #-----------------------------
    M = center[Y]
    dist = torch.abs(Z-M)
    dist = dist[dist>radius]
    counter=0
    if dist.size(0) == 0:
        loss=torch.tensor(0.0, dtype=Z.dtype, device=Z.device, requires_grad=True)
    else:
        if use_log == False:
            loss = torch.sum(dist)
        else:
            loss = torch.sum(torch.log(1+dist))
        counter=dist.size(0)
    #-----------------------------
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/Z.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%% https://arxiv.org/pdf/1711.09404.pdf
def dLdX_loss(X, L, norm):
    #L could be cross entropy loss
    # this should have been called before model(X) X.requires_grad=True
    X_grad=torch.autograd.grad(L, X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    if norm == 1:
        loss=torch.mean(torch.sum(X_grad.abs(), dim=1))
    elif norm == 2:
        loss=torch.mean(torch.sum(X_grad**2, dim=1))
    else:
        raise ValueError('norm should be 1 (L1) or 2 (L2)')
    return loss
#%%
def dZdX_loss_mask(model, X, Y, mask, norm=2, eps=1e-8):
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(Y.min().item(), Y.max().item()+1):
        Xn=X[Y==n]
        if Xn.size(0) <= 0:
            continue
        Xn=Xn.detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z
        #do not use Z.backward(): W.grad is updated by dZdW
        Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
        #--------------
        M=mask[Y==n] # 1: signal, 0: non-signal
        if norm == 1:
            loss=loss+torch.sum(Xn_grad[M<eps].abs())
        elif norm == 2:
            loss=loss+torch.sum(Xn_grad[M<eps]**2)
        else:
            raise ValueError('norm should be 1 (L1) or 2 (L2)')
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_loss1_fast(model, X, Y, mask=None, norm=2, alpha=1, eps=1e-8):
    # this should only be applied for correctly classified sampes
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) <= 1:
        Zy=Z
    else:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
    #--------------------------------
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #--------------------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)
    S=S/(torch.sum(S**2, dim=1, keepdim=True)+eps)    
    diff = X_grad-alpha*S
    #--------------------------------
    if norm == 1:
        loss=torch.mean(torch.sum(diff.abs(), dim=1))
    elif norm == 2:
        loss=torch.mean(torch.sum(diff**2, dim=1))
    else:
        raise ValueError('norm should be 1 (L1) or 2 (L2)')
    return loss
#%%
def dZdX_loss1(model, X, Y, mask=None, norm=2, alpha=1, eps=1e-8):
    # this should only be applied for correctly classified sampes
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(Y.min().item(), Y.max().item()+1):
        Xn=X[Y==n]
        if Xn.size(0) <= 0:
            continue
        Xn=Xn.detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z
        #do not use Z.backward(): W.grad is updated by dZdW
        Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
        #--------------
        Sn=Xn.detach()
        if mask is not None:
            M=mask[Y==n]
            Sn=Sn*M
        Sn=Sn.view(Sn.size(0), -1)
        Sn=Sn/(torch.sum(Sn**2, dim=1, keepdim=True)+eps)
        #--------------
        diff = Xn_grad-alpha*Sn
        if norm == 1:
            loss=loss+torch.sum(diff.abs())
        elif norm == 2:
            loss=loss+torch.sum(diff**2)
        else:
            raise ValueError('norm should be 1 (L1) or 2 (L2)')
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_loss1a(model, X, Y, mask=None, norm=2, eps=1e-8):
    X=X.detach()
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, X.size(0)):
        Xn=X[n:n+1].detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z
        #do not use Z.backward(): W.grad is updated by dZdW
        Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(-1)
        #---------------------
        Sn=Xn.detach()
        if mask is not None:
            M=mask[n:n+1]
            Sn=Sn*M
        Sn=Sn.view(-1)
        Sn=Sn/(torch.sum(Sn**2)+eps)
        #---------------------
        diff = Xn_grad-Sn
        region1=(M>eps)  #signal
        region2=(M<=eps) #non-signal
        if norm == 1:
            loss1=loss1+torch.mean(diff[region1].abs())
            loss2=loss2+torch.mean(diff[region2].abs())
        elif norm == 2:
            loss1=loss1+torch.sum(diff[region1]**2)
            loss2=loss2+torch.sum(diff[region2]**2)
        else:
            raise ValueError('norm should be 1 (L1) or 2 (L2)')
    loss1=loss1/X.size(0)
    loss2=loss2/X.size(0)
    return loss1, loss2
#%%
def dZdX_loss1b(model, X, Y, mask=None, norm=2, eps=1e-8):
    X=X.detach()
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(Y.min().item(), Y.max().item()+1):
        Xn=X[Y==n]
        if Xn.size(0) <= 0:
            continue
        Xn=Xn.detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z
        #do not use Z.backward(): W.grad is updated by dZdW
        Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
        #---------------------
        Sn=Xn.detach()
        if mask is not None:
            M=mask[Y==n]
            Sn=Sn*M
        Sn=Sn.view(Sn.size(0), -1)
        Sn=Sn/(torch.sum(Sn**2, dim=1, keepdim=True)+eps)
        #---------------------
        diff = Xn_grad-Sn
        region1=(M>eps)  #signal
        region2=(M<=eps) #non-signal
        if norm == 1:
            loss1=loss1+torch.sum(diff[region1].abs())
            loss2=loss2+torch.sum(diff[region2].abs())
        elif norm == 2:
            loss1=loss1+torch.sum(diff[region1]**2)
            loss2=loss2+torch.sum(diff[region2]**2)
        else:
            raise ValueError('norm should be 1 (L1) or 2 (L2)')
    loss1=loss1/X.size(0)
    loss2=loss2/X.size(0)
    return loss1, loss2
#%%
def dZdX_loss1cor(model, X, Y, mask=None, eps=1e-8):
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(Y.min().item(), Y.max().item()+1):
        Xn=X[Y==n]
        if Xn.size(0) <= 0:
            continue
        Xn=Xn.detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z
        #do not use Z.backward(): W.grad is updated by dZdW
        Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
        Xn_grad_norm=torch.sqrt(torch.sum(Xn_grad**2, dim=1, keepdim=True))
        Xn_grad_norm=Xn_grad_norm.detach()# must do this
        Xn_grad=Xn_grad/(Xn_grad_norm+eps)
        #--------------
        Sn=Xn.detach()
        if mask is not None:
            M=mask[Y==n]
            Sn=Sn*M
        Sn=Sn.view(Sn.size(0), -1)
        Sn_norm=torch.sqrt(torch.sum(Sn**2, dim=1, keepdim=True))
        Sn=Sn/(Sn_norm+eps)
        #--------------
        loss=loss+torch.sum((Xn_grad-Sn)**2)
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_loss1cor_obsv(model, X, Y, mask=None, eps=1e-8):
    #this is used to observe the corrolation between X and X.grad, not for training
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) <= 1:
        Zy=Z
    else:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
    #--------------------------------
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=False)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    X_grad_norm=torch.sqrt(torch.sum(X_grad**2, dim=1, keepdim=True))
    X_grad_norm=X_grad_norm.detach()
    X_grad=X_grad/(X_grad_norm+eps)
    #--------------------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)
    S_norm=torch.sqrt(torch.sum(S**2, dim=1, keepdim=True))
    S=S/(S_norm+eps)
    mean_cor = torch.mean(torch.sum(S*X_grad, dim=1))
    return mean_cor
#%%
def dZdX_loss1r(model, Xr, X, Y, mask=None, norm=2, eps=1e-8):
    # this should only be applied for correctly classified sampes
    Xr=Xr.detach()
    X=X.detach()
    Xr.requires_grad=True
    Zr=model(Xr)
    if len(Zr.size()) <= 1:
        Zry=Zr
    else:
        Zry=torch.gather(Zr, 1, Y.view(-1,1))
    #--------------------------------
    Xr_grad=torch.autograd.grad(Zry.sum(), Xr, create_graph=True)[0]
    Xr_grad=Xr_grad.view(Xr_grad.size(0), -1)
    #--------------------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)
    S=S/(torch.sum(S**2, dim=1, keepdim=True)+eps)    
    diff = Xr_grad-S
    #--------------------------------
    if norm == 1:
        loss=torch.mean(torch.sum(diff.abs(), dim=1))
    elif norm == 2:
        loss=torch.mean(torch.sum(diff**2, dim=1))
    else:
        raise ValueError('norm should be 1 (L1) or 2 (L2)')
    return loss
#%%
def dZdX_loss1rw(model, Xr, X, Y, norm=2):
    # this should only be applied for correctly classified sampes
    Xr=Xr.detach()
    X=X.detach()
    X.requires_grad=True
    Xr.requires_grad=True
    Z=model(X)
    Zr=model(Xr)
    if len(Z.size()) <= 1:
        Zy=Z
        Zry=Zr        
    else:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zry=torch.gather(Zr, 1, Y.view(-1,1))
    #--------------------------------
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    Xr_grad=torch.autograd.grad(Zry.sum(), Xr, create_graph=True)[0]
    Xr_grad=Xr_grad.view(Xr_grad.size(0), -1)
    diff = X_grad-Xr_grad
    #--------------------------------
    if norm == 1:
        loss=torch.mean(torch.sum(diff.abs(), dim=1))
    elif norm == 2:
        loss=torch.mean(torch.sum(diff**2, dim=1))
    else:
        raise ValueError('norm should be 1 (L1) or 2 (L2)')
    return loss
#%%
def dZdX_loss2(model, X, Y, mask=None, max_noise=1, alpha=1, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)# very important, otherwise Noise - Signal has wrong shape
    else:
        Zy=Z.abs()
    #---------------------         
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #--------------------- 
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)
    Signal = torch.sum(X_grad*S, dim=1) # z = w*x+b, assume b is 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0        
    #---------------------                
    Noise = torch.sum(X_grad**2, dim=1)*(max_noise**2)
    #---------------------
    Signal=Signal**2
    #----------------------------
    diff = Noise-alpha*Signal
    diff = diff[diff>0]
    if diff.size(0) > 0:
        loss = torch.sum(diff)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #----------------------------
    if reduction == 'counter':
        counter=diff.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise NotImplementedError("unkown reduction")  
    return loss  
#%%
def dZdX_loss2s(model, X, Y, mask=None, max_noise=1, alpha=0, eps=1e-8,
                log_NSR=False, detach_signal=False, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)# very important, otherwise Noise/(Signal+eps) has wrong shape
    else:
        Zy=Z.abs()
    #---------------------         
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #--------------------- 
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)
    Signal = torch.sum(X_grad*S, dim=1) # z = w*x+b, assume b is 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0        
    #---------------------                
    Noise = torch.sum(X_grad**2, dim=1)*(max_noise**2)
    #---------------------
    Signal=Signal**2
    #---------------------
    if detach_signal == True:
        Signal = Signal.detach()          
    #---------------------
    NSR = Noise/(Signal+eps)
    #---------------------------
    NSR = NSR[NSR>alpha]
    if NSR.size(0) > 0:
        if log_NSR == True:
            loss = sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)        
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #----------------------------
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise NotImplementedError("unkown reduction")  
    return loss     
#%%
def dZdX_loss2s_ref(model, X, Y, mask=None, max_noise=1, alpha=0, eps=1e-8,
                     log_NSR=False, detach_signal=False, reduction='mean'):    
    #this loss should only be applied for correctly classified samples
    #not for single output of binary classification (len(Z.size() <=1)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y.view(-1,1))
    Zy=Zy.view(-1)
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))        
    #---------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)             
    #--------------------------------
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for m in range(0, num_classes-1):
        ZSignal = Zy-Zother[:,m]  
        X_grad=torch.autograd.grad(ZSignal.sum(), X, create_graph=True)[0]
        X_grad=X_grad.view(X_grad.size(0), -1)
        #---------------------
        Signal = torch.sum(X_grad*S, dim=1) # z = w*x+b, assume b is 0
        #---------------------   
        Noise = torch.sum(X_grad**2, dim=1)*(max_noise**2)
        #---------------------
        Signal=Signal**2
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()
        #---------------------
        NSR = Noise/(Signal+eps)
        NSR = NSR[NSR>alpha]
        if NSR.size(0) >0:
            if log_NSR == True:
                loss += torch.sum(torch.log(1+NSR))
            else:
                loss += torch.sum(NSR)    
            counter+= NSR.size(0)
    #---------------------    
    if reduction == 'counter':
        loss=loss.sum()
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(Z.size(0)*(Z.size(1)-1))
    else:
        raise NotImplementedError("unkown reduction")  
    return loss 
#%%
def dZdX_loss2z(model, X, Y, max_noise=1, alpha=1, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)# very important, otherwise Noise/(Signal+eps) has wrong shape
    else:
        Zy=Z.abs()
    #-------------------------
    Signal = Zy # z = w'*x+b, b may not be 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: z should be >0
    #---------------------         
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------                
    Noise = torch.sum(X_grad**2, dim=1)*(max_noise**2)
    #---------------------
    Signal=Signal**2
    #---------------------
    diff = Noise-alpha*Signal
    diff = diff[diff>0]
    if diff.size(0) > 0:
        loss = torch.sum(diff)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------
    if reduction == 'counter':
        counter=diff.size(0)       
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_loss2zs_fast(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                      log_NSR=False, detach_signal=False, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)# very important, otherwise Noise/(Signal+eps) has wrong shape
    else:
        Zy=Z.abs()
    #-------------------------
    Signal = Zy # z = w'*x+b, b may not be 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: z should be >0
    #---------------------         
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------                
    Noise = torch.sum(X_grad**2, dim=1)*(max_noise**2)
    #---------------------
    Signal=Signal**2
    #---------------------
    if detach_signal == True:
        Signal = Signal.detach()          
    #---------------------
    NSR = Noise/(Signal+eps)
    #---------------------------
    NSR = NSR[NSR>alpha]
    if NSR.size(0) > 0:
        if log_NSR == True:
            loss = sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)        
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #----------------------------
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def dZdX_loss2zs(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                 log_NSR=False, detach_signal=False, reduction='mean'):
    #alpha is threshold of NSR
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(Y.min().item(), Y.max().item()+1):
        Xn=X[Y==n]
        if Xn.size(0) <= 0:
            continue
        Xn=Xn.detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z.abs()
        #---------------------
        Signal = Zn #z = w'*x+b, b may not be 0
        if len(Z.size()) > 1:
            Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: Zn should be >0    
        Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)            
        #---------------------                
        Noise = torch.sum(Xn_grad**2, dim=1)*(max_noise**2)
        #---------------------
        Signal=Signal**2
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()
        #---------------------
        NSR = Noise/(Signal+eps)
        NSR = NSR[NSR>alpha]
        if NSR.size(0) > 0:
            if log_NSR == True:
                loss = loss + torch.sum(torch.log(1+NSR))
            else:
                loss = loss + torch.sum(NSR)
            counter+=NSR.size(0)
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss  
#%%
def dZdX_loss2zs_ref(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                     log_NSR=False, detach_signal=False, reduction='mean'):    
    #this loss should only be applied for correctly classified samples
    #not for single output of binary classification (len(Z.size() <=1)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y.view(-1,1))
    Zy=Zy.view(-1)
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))        
    #--------------------------------
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for m in range(0, num_classes-1):
        Signal = Zy-Zother[:,m]  
        X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
        X_grad=X_grad.view(X_grad.size(0), -1)
        #---------------------   
        Noise = torch.sum(X_grad**2, dim=1)*(max_noise**2)
        #---------------------
        Signal=Signal**2
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()
        #---------------------
        NSR = Noise/(Signal+eps)
        NSR = NSR[NSR>alpha]
        if NSR.size(0) >0:
            if log_NSR == True:
                loss = loss+torch.sum(torch.log(1+NSR))
            else:
                loss = loss+torch.sum(NSR)    
            counter+= NSR.size(0)
    #---------------------    
    if reduction == 'counter':
        loss=loss.sum()
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(Z.size(0)*(Z.size(1)-1))
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def dZdX_loss2zs_ref_max(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                         log_NSR=False, detach_signal=False, reduction='mean'):    
    #this loss should only be applied for correctly classified samples
    #not for single output of binary classification (len(Z.size() <=1)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y.view(-1,1))
    Zy=Zy.view(-1)
    #--------------------------------
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))
    Zother_max = Zother.max(dim=1)[0]
    Signal=Zy-Zother_max
    #--------------------------------
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------                
    Noise = torch.sum(X_grad**2, dim=1)*(max_noise**2)
    #---------------------
    Signal=Signal**2
    #---------------------
    if detach_signal == True:
        Signal = Signal.detach()
    #---------------------
    NSR = Noise/(Signal+eps)
    NSR = NSR[NSR>alpha]
    if NSR.size(0) > 0:
        if log_NSR == True:
            loss = torch.sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)    
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------    
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/Z.size(0)
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def dZdX_loss3(model, X, Y, mask=None, max_noise=1, alpha=1, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)
    else:
        Zy=Z.abs()
    #---------------------         
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)
    Signal = torch.sum(X_grad*S, dim=1) # z = w*x+b, assume b is 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0   
    #---------------------
    Noise = torch.sum(X_grad.abs(), dim=1)*max_noise
    #---------------------
    diff = Noise-alpha*Signal
    diff = diff[diff>0]
    if diff.size(0) > 0:
        loss = torch.sum(diff)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------
    if reduction == 'counter':
        counter=diff.size(0)
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
#%%
def dZdX_loss3s(model, X, Y, mask=None, max_noise=1, alpha=0, eps=1e-8,
                log_NSR=False, detach_signal=False, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)
    else:
        Zy=Z.abs()
    #---------------------         
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)
    Signal = torch.sum(X_grad*S, dim=1) # z = w*x+b, assume b is 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0   
    #---------------------
    Noise = torch.sum(X_grad.abs(), dim=1)*max_noise
    #---------------------
    NSR = Noise/(Signal+eps)
    NSR = NSR[NSR>alpha]
    if NSR.size(0) >0:
        if log_NSR == True:
            loss = torch.sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------    
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss.mean()
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def dZdX_loss3s_ref_max(model, X, Y, mask=None, max_noise=1, alpha=0, eps=1e-8,
                        log_NSR=False, detach_signal=False, reduction='mean'):    
    #this loss should only be applied for correctly classified samples
    #not for single output of binary classification (len(Z.size() <=1)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y.view(-1,1))
    Zy=Zy.view(-1)
    #--------------------------------
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))       
    Zother_max = Zother.max(dim=1)[0]
    ZSignal=Zy-Zother_max
    #--------------------------------
    X_grad=torch.autograd.grad(ZSignal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)             
    Signal = torch.sum(X_grad*S, dim=1) # z = w*x+b, assume b is 0
    #---------------------                
    Noise = torch.sum(X_grad.abs(), dim=1)*max_noise
    #---------------------
    if detach_signal == True:
        Signal = Signal.detach()
    #---------------------
    NSR = Noise/(Signal+eps)
    NSR = NSR[NSR>alpha]
    if NSR.size(0) >0:
        if log_NSR == True:
            loss = torch.sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------    
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss.mean()
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def dZdX_loss3z(model, X, Y, mask, max_noise=1, alpha=1, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X, mask)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)# very important, otherwise Noise - Signal has wrong shape
    else:
        Zy=Z.abs()
    #-------------------------
    Signal = Zy # z = w'*x+b, b may not be 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: z should be >0
    #---------------------         
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    Noise = torch.sum(X_grad.abs(), dim=1)*max_noise
    #---------------------
    diff = Noise-alpha*Signal
    diff = diff[diff>0]
    if diff.size(0) > 0:
        loss = torch.sum(diff)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------
    if reduction == 'counter':
        counter=diff.size(0)
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
#%%
def dZdX_loss3zs_fast(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                      log_NSR=False, detach_signal=False, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)# very important, otherwise Noise/(Signal+eps) has wrong shape
    else:
        Zy=Z.abs()
    #-------------------------
    Signal = Zy # z = w'*x+b, b may not be 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: z should be >0
    #---------------------         
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    Noise = torch.sum(X_grad.abs(), dim=1)*max_noise
    #---------------------
    if detach_signal == True:
        Signal = Signal.detach()          
    #---------------------
    NSR = Noise/(Signal+eps)
    #---------------------------
    NSR = NSR[NSR>alpha]
    if NSR.size(0) > 0:
        if log_NSR == True:
            loss = sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)        
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #----------------------------
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise NotImplementedError("unkown reduction")  
    return loss     
                
    '''
    Mask=torch.zeros_like(NSR)
    Mask[NSR>alpha]=1
    NSR = NSR*Mask
    counter=Mask.sum()
    loss = NSR
    if log_NSR == True:
        loss = torch.log(1+loss)
    #---------------------
    if reduction == 'counter':
        loss=loss.sum()
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss.mean()
    elif reduction is None:
        pass
    else:
        raise NotImplementedError("unkown reduction")  
    return loss     
    '''
#%% faster version of loss3zs
def dZdX_NSR(model, X, Y, mask, noise_norm=1, norm_type=np.inf,
             log_NSR=False, reduction='mean', model_eval=False, alpha=0, eps=1e-8):
    # this shoud only be applied for correctly classified samples with MSE or MAE loss
    train_mode=model.training# record the mode
    if model_eval == True and train_mode == True:
        model.eval()
    #--------------------------------------------
    X=X.detach()
    X.requires_grad=True
    Z=model(X, mask)
    # Z.shape is (N,C), Y.shape is (N,)
    if len(Z.size()) > 1:
        Zy=Zy=Z[torch.arange(Y.shape[0]), Y]
    else:
        Zy=Z.abs()
    #-------------------------
    Signal = Zy # z = w'*x+b, b may not be 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: z should be >0
    #---------------------
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    if norm_type == np.inf:
        p_norm=1
    elif norm_type == 2:
        p_norm=2
    Noise = noise_norm*torch.norm(X_grad, p=p_norm, dim=1)
    #---------------------
    NSR = Noise/(Signal+eps)
    #---------------------------
    NSR = NSR[NSR>alpha]
    if NSR.size(0) > 0:
        if log_NSR == True:
            loss = sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #--------------------------------------------
    if train_mode == True and model.training == False:
        model.train()
    #--------------------------------------------
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    elif reduction == 'sum':
        pass
    else:
        raise NotImplementedError("unkown reduction")
    return loss
#%% in theory, dZdX_loss3zs is the same as dZdX_loss3zs_fast and dZdX_loss3zs_slow
def dZdX_loss3zs(model, X, Y, mask, max_noise=1, alpha=0, eps=1e-8,
                 log_NSR=False, detach_signal=False, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(Y.min().item(), Y.max().item()+1):
        Xn=X[Y==n]
        if Xn.size(0) <= 0:
            continue
        Xn=Xn.detach()
        Xn.requires_grad=True
        Z=model(Xn, mask[Y==n])
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z.abs()
        #---------------------
        Signal = Zn # z = w'*x+b, b may not be 0
        if len(Z.size()) > 1: # should be Z
            Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: Zn should be >0
            #So, this loss is incompatible with cross entropy loss
        #---------------------
        Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
        #---------------------
        Noise = torch.sum(Xn_grad.abs(), dim=1)*max_noise
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()       
        #---------------------
        NSR = Noise/(Signal+eps)
        NSR = NSR[NSR>alpha]
        if NSR.size(0) > 0:
            if log_NSR == True:
                loss = loss + torch.sum(torch.log(1+NSR))
            else:
                loss = loss + torch.sum(NSR)
            counter+=NSR.size(0)
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_loss3zs_slow(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                      log_NSR=False, detach_signal=False, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, X.size(0)):
        Xn=X[n:n+1].detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[0,Y[n]]
        else:
            Zn=Z.abs()        
        #---------------------
        Signal = Zn # z = w'*x+b, b may not be 0
        if Signal <= 0:
            continue
        #---------------------
        Xn_grad=torch.autograd.grad(Signal, Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(-1)
        #---------------------
        Noise = torch.sum(Xn_grad.abs())*max_noise
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()       
        #---------------------
        NSR = Noise/(Signal+eps)
        if NSR > alpha:
            if log_NSR == True:
                loss = loss + torch.log(1+NSR)
            else:
                loss = loss + NSR
            counter+=1
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss   
#%%
def dZdX_loss3zs_ref(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                     log_NSR=False, detach_signal=False, reduction='mean'):    
    #this loss should only be applied for correctly classified samples
    #not for single output of binary classification (len(Z.size() <=1)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y.view(-1,1))
    Zy=Zy.view(-1)
    #--------------------------------
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))        
    #--------------------------------
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for m in range(0, num_classes-1):
        Signal = Zy-Zother[:,m]  
        X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
        X_grad=X_grad.view(X_grad.size(0), -1)
        #---------------------                
        Noise = torch.sum(X_grad.abs(), dim=1)*max_noise
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()
        #---------------------
        NSR = Noise/(Signal+eps)
        NSR = NSR[NSR>alpha]
        if NSR.size(0) >0:
            if log_NSR == True:
                loss = loss+torch.sum(torch.log(1+NSR))
            else:
                loss = loss+torch.sum(NSR)    
            counter+= NSR.size(0)
    #---------------------    
    if reduction == 'counter':
        loss=loss.sum()
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(Z.size(0)*(Z.size(1)-1))
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def dZdX_loss3zs_ref_max(model, X, Y, max_noise=1, alpha=0, eps=1e-8,
                         log_NSR=False, detach_signal=False, reduction='mean'):    
    #this loss should only be applied for correctly classified samples
    #not for single output of binary classification (len(Z.size() <=1)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    num_classes=Z.size(1)
    Zy=torch.gather(Z, 1, Y.view(-1,1))
    Zy=Zy.view(-1)
    #--------------------------------
    idxTable=torch.zeros((Z.size(0), num_classes-1), dtype=torch.int64)
    idxlist = torch.arange(0, num_classes, dtype=torch.int64)
    Ycpu=Y.cpu()
    for n in range(Z.size(0)):
        idxTable[n]=idxlist[idxlist!=Ycpu[n]]
    Zother=torch.gather(Z, 1, idxTable.to(Z.device))
    Zother_max = Zother.max(dim=1)[0]
    Signal=Zy-Zother_max
    #--------------------------------
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------                
    Noise = torch.sum(X_grad.abs(), dim=1)*max_noise
    #---------------------
    if detach_signal == True:
        Signal = Signal.detach()
    #---------------------
    NSR = Noise/(Signal+eps)
    NSR = NSR[NSR>alpha]
    if NSR.size(0) >0:
        if log_NSR == True:
            loss = torch.sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------    
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/Z.size(0)
    else:
        raise NotImplementedError("unkown reduction")  
    return loss
#%%
def dZdX_loss3zs_Xn(model, X, Xn, Y, alpha=0, eps=1e-8,
                    log_NSR=False, detach_signal=False, reduction='mean'):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    Xn=Xn.detach()
    noise=(Xn-X).detach()
    Xn.requires_grad=True
    Zn=model(Xn)
    if len(Zn.size()) > 1:
        Zyn=torch.gather(Zn, 1, Y.view(-1,1))
        Zyn=Zyn.view(-1)# very important, otherwise Noise/(Signal+eps) has wrong shape
    else:
        Zyn=Zn
    #-----------------------                 
    if len(Zn.size()) > 1:
        Zyn = Zyn*(Zyn>0).to(Zyn.dtype) # multi-class: Zyn should be >0
    #---------------------         
    Xn_grad=torch.autograd.grad(Zyn.sum(), Xn, create_graph=True)[0]
    Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
    #-----------------------
    #Noise is output 'noise'
    noise=noise.view(noise.size(0),-1)    
    Noise=torch.sum(Xn_grad*noise, dim=1)    
    #Signal=Zyn-Noise # z = w'*x + w'*n + b
    Signal=Zyn
    #-----------------------
    Signal=Signal.abs()
    Noise=Noise.abs()
    #-----------------------
    if detach_signal == True:
        Signal = Signal.detach()          
    #-----------------------
    NSR = Noise/(Signal+eps)
    #-----------------------
    NSR = NSR[NSR>alpha]
    if NSR.size(0) > 0:
        if log_NSR == True:
            loss = torch.sum(torch.log(1+NSR))
        else:
            loss = torch.sum(NSR)        
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #-----------------------
    if reduction == 'counter':
        counter=NSR.size(0)
        if counter > 1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise NotImplementedError("unkown reduction")  
    return loss      
#%%
def dZdX_loss3zs2zs(model, X, Y, beta=1, eps=1e-8, log_NSR=False, detach_signal=False):
    # this shoud only be applied for correctly classified samples using MSE loss
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)# very important, otherwise Noise/(Signal+eps) has wrong shape
    else:
        Zy=Z.abs()
    #-------------------------
    Signal = Zy # z = w'*x+b, b may not be 0
    if len(Z.size()) > 1:
        Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: z should be >0
    #---------------------         
    X_grad=torch.autograd.grad(Signal.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    NoiseA = torch.sum(X_grad.abs(), dim=1)
    NoiseB = torch.sum(X_grad**2, dim=1)
    #---------------------
    if detach_signal == True:
        Signal = Signal.detach()          
    #---------------------
    NSR = NoiseA/(Signal+eps)+beta*NoiseB/(Signal**2+eps)
    #---------------------------
    if log_NSR == True:
        loss = sum(torch.log(1+NSR))
    else:
        loss = torch.sum(NSR)        
    #----------------------------
    loss=loss/X.size(0)
    return loss                
#%%
def dZdX_wx_constant(model, X, Y, mask=None, alpha=1):
    #z=w*x+b, s.t. w*x=alpha
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) <= 1:
        Zy=Z
    else:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
    #----------------------------------
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    S=X.detach()
    if mask is not None:
        S=S*mask
    S=S.view(S.size(0), -1)             
    Signal = torch.sum(X_grad*S, dim=1)
    loss = torch.mean((Signal-alpha)**2)
    return loss
#%%
def dZdX_minimize_bias(model, X, Y, mask=None, alpha=1, reduction='mean'):
    #minimize b in z=w*x+b such that |b|<=alpha
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(Y.min().item(), Y.max().item()+1):
        Xn=X[Y==n]
        if Xn.size(0) <= 0:
            continue
        counter+=1
        Xn=Xn.detach()
        Xn.requires_grad=True
        Z=model(Xn)
        if len(Z.size()) > 1:
            Zn=Z[:,n]
        else:
            Zn=Z
        Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
        #--------------
        Sn=Xn.detach()
        if mask is not None:
            M=mask[Y==n]
            Sn=Sn*M
        Sn=Sn.view(Sn.size(0), -1)
        #--------------            
        Signal = torch.sum(Xn_grad*Sn, dim=1)
        Bias = Zn-Signal  
        Bias = Bias[Bias.abs()>alpha]
        if Bias.size(0)>0:
            loss = loss + torch.sum(Bias**2)
            counter+=Bias.size(0)
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_norm(model, X, Y, norm, reduction='mean'):
    #z=w*x+b, return L1 norm ||w|| or L2 norm ||w||^2
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)
    else:
        Zy=Z
    #---------------------         
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    if reduction == 'mean':
        loss=torch.mean(torch.norm(X_grad, p=norm, dim=1))
    elif reduction == 'sum':
        loss=torch.sum(torch.norm(X_grad, p=norm, dim=1))
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_L2norm_constant(model, X, Y, alpha=1):
    #z=w*x+b, s.t. L2 norm ||w|| = alpha
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)
    else:
        Zy=Z
    #---------------------         
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)
    #---------------------
    loss=torch.mean((torch.sum(X_grad**2, dim=1)-alpha)**2)
    return loss
#%%
def dZdX_L2norm_max(model, X, Y, alpha=1, reduction='mean'):
    #z=w*x+b, s.t. L2 norm ||w|| <= alpha
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    if len(Z.size()) > 1:
        Zy=torch.gather(Z, 1, Y.view(-1,1))
        Zy=Zy.view(-1)
    else:
        Zy=Z
    #---------------------         
    X_grad=torch.autograd.grad(Zy.sum(), X, create_graph=True)[0]
    X_grad=X_grad.view(X_grad.size(0), -1)    
    L2norm_sq = torch.sum(X_grad**2, dim=1)
    L2norm_sq = L2norm_sq[L2norm_sq>alpha**2]
    if L2norm_sq.size(0) >0:
        loss=torch.sum(L2norm_sq)
    else:
        loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    #---------------------
    if reduction == 'counter':
        counter=L2norm_sq.size(0)
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss

#%%
def dZdX_jacob(model, X, Y, mask, norm, reduction='mean', return_Z=False):
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    X=X.detach()
    X.requires_grad=True
    Z=model(X, mask)
    Z=Z.view(Z.size(0), -1)
    #X = X[mask==1]
    for m in range(0, Z.size(1)):
        Zm=Z[:,m].sum()
        X_grad=torch.autograd.grad(Zm, X,create_graph=True)[0]
        X_grad=X_grad.view(X_grad.size(0), -1)
        Lp_norm=torch.norm(X_grad, p=norm, dim=1)
        loss=loss+torch.sum(torch.pow(Lp_norm, 2))
    loss = torch.pow(loss, 0.5)
    if reduction=='mean':
        loss=loss/(Z.size(0)*Z.size(1))
    elif reduction=='sum':
        pass
    else:
        raise ValueError('unkown reduction')
    if return_Z == False:
        return loss
    else:
        return loss, Z
    
#%%
def dZdX_jacob_old(model, X, Y, mask, norm, reduction='mean', return_Z=False):
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    X=X.detach()
    X.requires_grad=True
    Z=model(X, mask)
    Z=Z.view(Z.size(0), -1)
    #X = X[mask==1]
    for m in range(0, Z.size(1)):
        Zm=Z[:,m].sum()
        X_grad=torch.autograd.grad(Zm, X,create_graph=True)[0]
        X_grad=X_grad.view(X_grad.size(0), -1)
        Lp_norm=torch.norm(X_grad, p=norm, dim=1)
        loss=loss+torch.sum(Lp_norm)
    if reduction=='mean':
        loss=loss/(Z.size(0)*Z.size(1))
    elif reduction=='sum':
        pass
    else:
        raise ValueError('unkown reduction')
    if return_Z == False:
        return loss
    else:
        return loss, Z  

#%%
def dZdX_jacob_orthogonal(model, X):
    #not for single output of binary classification (len(Z.size() <=1)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    loss=0
    for i in range(0, Z.size(1)):
        Zi=Z[:,i].sum()
        grad_i=torch.autograd.grad(Zi, X, create_graph=True)[0]
        grad_i=grad_i.view(grad_i.size(0),-1)
        for j in range(i+1, Z.size(1)):
            Zj=Z[:,j].sum()
            grad_j=torch.autograd.grad(Zj, X, create_graph=True)[0]
            grad_j=grad_j.view(grad_j.size(0),-1)
            loss=loss+ torch.sum(torch.abs(torch.sum(grad_i*grad_j, dim=1)))
    counter=Z.size(0)*Z.size(1)*(Z.size(1)-1)/2
    loss=loss/counter
    return loss
#%%
def kernel_loss_of_module(Weight, dZdX_loss):
    #Weight: (K, L),  or (K, C, H, W) weight of a module
    #dZdX_loss may be loss2 from dZdX_loss2
    dLdW = torch.autograd.grad(dZdX_loss, Weight, retain_graph=True)[0].detach()
    temp = dLdW*Weight
    temp = temp.view(temp.size(0), -1)
    loss = torch.sum(torch.sum(temp, dim=1)**2)
    return loss
#%%
def kernel_loss(model, dZdX_loss):
    loss=0
    for i in range(0, len(model.E)):
        loss=loss+kernel_loss_of_module(model.E[i].weight, dZdX_loss)
    for i in range(0, len(model.C)):
        loss=loss+kernel_loss_of_module(model.C[i].weight, dZdX_loss)
    return loss
#%%
def link_loss_of_module(Weight, dZdX_loss):
    #Weight: (K, L),  weight of a module (Linear, Conv1d, etc...)
    #dZdX_loss may be loss2 from dZdX_loss2
    dLdW = torch.autograd.grad(dZdX_loss, Weight, retain_graph=True)[0].detach()
    temp = dLdW*Weight
    loss = torch.sum(temp**2)
    return loss
#%%
def link_loss(model, dZdX_loss):
    loss=0
    for i in range(0, len(model.E)):
        loss=loss+link_loss_of_module(model.E[i].weight, dZdX_loss)
    for i in range(0, len(model.C)):
        loss=loss+link_loss_of_module(model.C[i].weight, dZdX_loss)
    return loss
#%%