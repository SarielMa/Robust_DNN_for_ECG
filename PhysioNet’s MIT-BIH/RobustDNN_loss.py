import torch
import torch.nn.functional as nnF
import warnings
#%%
def HingeLoss(Z, Y, margin, within_margin, use_log=False, reduction='mean'):
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
    loss=torch.tensor(0.0, dtype=Z.dtype, device=Z.device, requires_grad=True)
    counter=0
    num_classes=Z.size(1)
    for n in range(0, num_classes):
        Zn=Z[Y==n]
        Zy=Zn[:,n].view(-1,1)
        Zother=torch.cat([Zn[:,0:n], Zn[:,n+1:]], dim=1)
        temp = margin-Zy+Zother
        if within_margin == False:
            temp=temp[temp>0]
        else:
            temp=temp[(temp>0)&(temp<2*margin)]
        if temp.size(0) > 0:
            if use_log == False:
                loss=loss+torch.sum(temp)
            else:
                loss=loss+torch.sum(torch.log(1+temp))
            counter+=temp.size(0)
    #----------------------------
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(Z.size(0)*Z.size(1))
    else:
        raise ValueError('unkown reduction')
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
def dZdX_loss_mask(model, X, Y, num_classes, mask, norm=2, eps=1e-8):
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
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
def dZdX_loss1(model, X, Y, num_classes, mask=None, norm=2, eps=1e-8):
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
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
            diff = Xn_grad-Sn
            if norm == 1:
                loss=loss+torch.sum(diff.abs())
            elif norm == 2:
                loss=loss+torch.sum(diff**2)
            else:
                raise ValueError('norm should be 1 (L1) or 2 (L2)')
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_loss1a(model, X, Y, num_classes, mask=None, norm=2, eps=1e-8):
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
def dZdX_loss1b(model, X, Y, num_classes, mask=None, norm=2, eps=1e-8):
    X=X.detach()
    loss1=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    loss2=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
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
def dZdX_loss1cor(model, X, Y, num_classes, mask=None, eps=1e-8):
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
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
def dZdX_loss1cor_obsv(model, X, Y, num_classes, mask=None, eps=1e-8):
    #this is used to observe the corrolation between X and X.grad, not for training
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
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
            Xn_grad_norm=Xn_grad_norm.detach()
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
            loss = loss + torch.sum(Sn*Xn_grad)
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_loss1r(model, Xr, X, Y, num_classes, mask=None, norm=2, eps=1e-8):
    #Xr: random noise + X
    Xr=Xr.detach()
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xrn=Xr[Y==n].detach()
        Xn=X[Y==n].detach()
        if Xn.size(0) > 0:
            Xrn.requires_grad=True
            Z=model(Xrn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            #do not use Z.backward(): W.grad is updated by dZdW
            Xrn_grad=torch.autograd.grad(Zn.sum(), Xrn, create_graph=True)[0]
            Xrn_grad=Xrn_grad.view(Xrn_grad.size(0), -1)
            #---------------------
            Sn=Xn.detach()
            if mask is not None:
                M=mask[Y==n]
                Sn=Sn*M
            Sn=Sn.view(Sn.size(0), -1)
            Sn=Sn/(torch.sum(Sn**2, dim=1, keepdim=True)+eps)
            #---------------------
            diff = Xrn_grad-Sn
            if norm == 1:
                loss=loss+torch.sum(diff.abs())
            elif norm == 2:
                loss=loss+torch.sum(diff**2)
            else:
                raise ValueError('norm should be 1 (L1) or 2 (L2)')
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_loss2(model, X, Y, num_classes, mask=None, max_noise=1, alpha=1, reduction='mean'):
    #max_noise: L2 norm
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------
            Sn=Xn.detach()
            if mask is not None:
                M=mask[Y==n]
                Sn=Sn*M
            Sn=Sn.view(Sn.size(0), -1)
            #---------------------
            Signal = torch.sum(Xn_grad*Sn, dim=1) # z = w*x+b, assume b is 0
            if len(Z.size()) > 1:
                Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0            
            Signal = Signal**2 
            #---------------------
            Noise = torch.sum(Xn_grad**2, dim=1)*(max_noise**2)
            #---------------------
            if alpha==1:
                loss = loss+torch.sum(Noise-Signal)#Noise>Signal always
                counter+=Signal.size(0)
            else:
                diff = Noise-Signal*(alpha**2)
                diff = diff[diff>0]
                if diff.size(0) > 0:
                    loss = loss + torch.sum(diff)
                    counter+=diff.size(0)                
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_loss2s(model, X, Y, num_classes, mask=None, max_noise=1, alpha=0, eps=1e-8,
                log_NSR=False, detach_signal=False, reduction='mean'):
    #alpha is threshold of NSR
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------
            Sn=Xn.detach()
            if mask is not None:
                M=mask[Y==n]
                Sn=Sn*M
            Sn=Sn.view(Sn.size(0), -1)            
            #---------------------
            Signal = torch.sum(Xn_grad*Sn, dim=1) # z = w*x+b, assume b is 0
            if len(Z.size()) > 1:
                Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0           
            Signal = Signal**2    
            #---------------------
            if detach_signal == True:
                Signal = Signal.detach()
            #---------------------
            Noise = torch.sum(Xn_grad**2, dim=1)*(max_noise**2)
            #---------------------
            NSR = Noise/(Signal+eps)
            #print(NSR)
            NSR = NSR[NSR>(alpha**2)]
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
def dZdX_loss2z(model, X, Y, num_classes, max_noise=1, alpha=1, reduction='mean'):
    #alpha is threshold of NSR
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            #---------------------
            Signal = Zn #z = w'*x+b, b may not be 0
            if len(Z.size()) > 1:
                Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: Zn should be >0    
            Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------       
            Signal=Signal**2
            #---------------------
            Noise = torch.sum(Xn_grad**2, dim=1)*max_noise
            if alpha==1:            
                loss = loss + torch.sum(Noise-Signal)
                counter+=Signal.size(0)
            else:
                diff = Noise-Signal*(alpha**2)
                diff = diff[diff>0]
                if diff.size(0) > 0:
                    loss = loss + torch.sum(diff)
                    counter+=diff.size(0)       
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_loss2zs(model, X, Y, num_classes, max_noise=1, alpha=0, eps=1e-8,
                 log_NSR=False, detach_signal=False, reduction='mean'):
    #alpha is threshold of NSR
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            #---------------------
            Signal = Zn #z = w'*x+b, b may not be 0
            if len(Z.size()) > 1:
                Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: Zn should be >0    
            Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)            
            #---------------------
            Signal=Signal**2
            #---------------------
            if detach_signal == True:
                Signal = Signal.detach()
            #---------------------                
            Noise = torch.sum(Xn_grad**2, dim=1)*(max_noise**2)
            #---------------------
            NSR = Noise/(Signal+eps)
            NSR = NSR[NSR>(alpha**2)]
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
def dZdX_loss2s_ref(model, X, Y, num_classes, mask=None, max_noise=1, alpha=0, eps=1e-8,
                    log_NSR=False, detach_signal=False, reduction='mean'):    
    #not for single output of binary classification (len(Zn.size() <=1)
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n].detach()
        if Xn.size(0) <= 0:
            continue
        idx_other=list(range(0, n))+list(range(n+1,num_classes))        
        Xn.requires_grad=True
        Z=model(Xn)      
        Zn=Z[:,n]
        #---------------------
        Sn=Xn.detach()
        if mask is not None:
            M=mask[Y==n]
            Sn=Sn*M
        Sn=Sn.view(Sn.size(0), -1)            
        #---------------------
        for m in idx_other:
            ZSignal = Zn-Z[:,m]
            Xn_grad=torch.autograd.grad(ZSignal.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------
            Signal = torch.sum(Xn_grad*Sn, dim=1) # z = w*x+b, assume b is 0
            #---------------------
            if detach_signal == True:
                Signal = Signal.detach()
            #---------------------
            Signal=Signal**2
            #---------------------                
            Noise = torch.sum(Xn_grad**2, dim=1)*(max_noise**2)
            #---------------------
            NSR = Noise/(Signal+eps)
            NSR = NSR[NSR>alpha]
            if NSR.size(0) > 0:
                if log_NSR == True:
                    loss = loss + torch.sum(torch.log(1+NSR))
                else:
                    loss = loss + torch.sum(NSR)
                counter+=NSR.size(0)
        #---------------------           
    if reduction == 'counter':
        if counter>0:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(X.size(0)*(num_classes-1))
    else:
        raise ValueError('unkown reduction')
    return loss 
#%%
def dZdX_loss2zs_ref(model, X, Y, num_classes, max_noise=1, alpha=0, eps=1e-8,
                     log_NSR=False, detach_signal=False, reduction='mean'):    
    #not for single output of binary classification (len(Zn.size() <=1)
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n].detach()
        if Xn.size(0) <= 0:
            continue
        idx_other=list(range(0, n))+list(range(n+1,num_classes))   
        Xn.requires_grad=True
        Z=model(Xn)      
        Zn=Z[:,n]
        #---------------------
        for m in idx_other:
            Signal = Zn-Z[:,m]
            Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)        
            #---------------------
            if detach_signal == True:
                Signal = Signal.detach()
            #---------------------
            Signal=Signal**2
            #---------------------                
            Noise = torch.sum(Xn_grad**2, dim=1)*(max_noise**2)
            #---------------------
            NSR = Noise/(Signal+eps)
            NSR = NSR[NSR>alpha]
            if NSR.size(0) > 0:
                if log_NSR == True:
                    loss = loss + torch.sum(torch.log(1+NSR))
                else:
                    loss = loss + torch.sum(NSR)
                counter+=NSR.size(0)
        #---------------------           
    if reduction == 'counter':
        if counter>0:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(X.size(0)*(num_classes-1))
    else:
        raise ValueError('unkown reduction')
    return loss    
#%%
def dZdX_loss2zs_ref_max(model, X, Y, num_classes, max_noise=1, alpha=0, eps=1e-8,
                         log_NSR=False, detach_signal=False, reduction='mean'):    
    #not for single output of binary classification (len(Zn.size() <=1)
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n].detach()
        if Xn.size(0) <= 0:
            continue
        idx_other=list(range(0, n))+list(range(n+1,num_classes))
        Xn.requires_grad=True
        Z=model(Xn)      
        Zn=Z[:,n]
        Zother=Z[:,idx_other]
        Zother_max=torch.max(Zother, dim=1)[0]
        Zother_max=Zother_max.view(Zn.size())
        #---------------------
        Signal = Zn-Zother_max
        Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)        
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()
        #---------------------
        Signal=Signal**2
        #---------------------                
        Noise = torch.sum(Xn_grad**2, dim=1)*(max_noise**2)
        #---------------------
        NSR = Noise/(Signal+eps)
        NSR = NSR[NSR>alpha]
        if NSR.size(0) > 0:
            if log_NSR == True:
                loss = loss + torch.sum(torch.log(1+NSR))
            else:
                loss = loss + torch.sum(NSR)
            counter+=NSR.size(0)
        #---------------------           
    if reduction == 'counter':
        if counter>0:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss 
#%%
def dZdX_loss3(model, X, Y, num_classes, mask=None, max_noise=1, alpha=1, reduction='mean'):
    #max_noise: L_inf norm
    #alpha is threshold of NSR
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------
            Sn=Xn.detach()
            if mask is not None:
                M=mask[Y==n]
                Sn=Sn*M
            Sn=Sn.view(Sn.size(0), -1)
            #---------------------
            Signal = torch.sum(Xn_grad*Sn, dim=1) # z = w*x+b, assume b is 0
            if len(Z.size()) > 1:
                Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0   
            #---------------------
            Noise = torch.sum(Xn_grad.abs(), dim=1)*max_noise
            #---------------------        
            diff = Noise-alpha*Signal.abs()
            diff = diff[diff>0]
            if diff.size(0) > 0:
                loss = loss + torch.sum(diff)
                counter+=diff.size(0)
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_loss3s(model, X, Y, num_classes, mask=None, max_noise=1, alpha=0, eps=1e-8,
                log_NSR=False, detach_signal=False, reduction='mean'):
    #alpha is threshold of NSR
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------
            Sn=Xn.detach()
            if mask is not None:
                M=mask[Y==n]
                Sn=Sn*M
            Sn=Sn.view(Sn.size(0), -1)
            #---------------------
            Signal = torch.sum(Xn_grad*Sn, dim=1) # z = w*x+b, assume b is 0
            if len(Z.size()) > 1:
                Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: w*x should be >0
            #---------------------
            if detach_signal == True:
                Signal = Signal.detach()
            Noise = torch.sum(Xn_grad.abs(), dim=1)*max_noise
            #---------------------
            NSR = Noise/(Signal.abs()+eps)
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
def dZdX_loss3z(model, X, Y, num_classes, max_noise=1, alpha=1, reduction='mean'):
    #alpha is threshold of NSR
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            #---------------------    
            Signal = Zn # z = w*x+b, b may not be 0
            if len(Zn.size()) > 1:
                Signal = Signal*(Signal>0).to(Signal.dtype) # multi-class: Zn should be >0
            Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------
            Noise = torch.sum(Xn_grad.abs(), dim=1)*max_noise
            #---------------------
            diff = Noise-alpha*Signal.abs()
            diff = diff[diff>0]
            if diff.size(0) > 0:
                loss = loss + torch.sum(diff)
                counter+=diff.size(0)
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_loss3zs(model, X, Y, num_classes, max_noise=1, alpha=0, eps=1e-8,
                 log_NSR=False, detach_signal=False, reduction='mean'):
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            #---------------------
            Signal = Zn # z = w'*x+b, b may not be 0
            if len(Zn.size()) > 1:
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
            NSR = Noise/(Signal.abs()+eps)
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
def dZdX_loss3s_ref(model, X, Y, num_classes, mask=None, max_noise=1, alpha=0, eps=1e-8,
                    log_NSR=False, detach_signal=False, reduction='mean'):    
    #not for single output of binary classification (len(Zn.size() <=1)
    #this loss should only be applied for correctly classified samples
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n].detach()
        if Xn.size(0) <= 0:
            continue
        idx_other=list(range(0, n))+list(range(n+1,num_classes))        
        Xn.requires_grad=True
        Z=model(Xn)      
        Zn=Z[:,n]
        #---------------------
        Sn=Xn.detach()
        if mask is not None:
            M=mask[Y==n]
            Sn=Sn*M
        Sn=Sn.view(Sn.size(0), -1)            
        #---------------------
        for m in idx_other:
            ZSignal = Zn-Z[:,m]
            Xn_grad=torch.autograd.grad(ZSignal.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            #---------------------
            Signal = torch.sum(Xn_grad*Sn, dim=1) # z = w*x+b, assume b is 0
            #---------------------
            if detach_signal == True:
                Signal = Signal.detach()
            #---------------------                
            Noise = torch.sum(Xn_grad.abs(), dim=1)*max_noise
            #---------------------
            NSR = Noise/(Signal+eps)
            NSR = NSR[NSR>alpha]
            if NSR.size(0) > 0:
                if log_NSR == True:
                    loss = loss + torch.sum(torch.log(1+NSR))
                else:
                    loss = loss + torch.sum(NSR)
                counter+=NSR.size(0)
        #---------------------           
    if reduction == 'counter':
        if counter>0:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(X.size(0)*(num_classes-1))
    else:
        raise ValueError('unkown reduction')
    return loss 
#%%
def dZdX_loss3zs_ref(model, X, Y, num_classes, max_noise=1, alpha=0, eps=1e-8,
                     log_NSR=False, detach_signal=False, reduction='mean'):    
    #not for single output of binary classification (len(Zn.size() <=1)
    #this loss should only be applied for correctly classified samples
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n].detach()
        if Xn.size(0) <= 0:
            continue
        idx_other=list(range(0, n))+list(range(n+1,num_classes))        
        Xn.requires_grad=True
        Z=model(Xn)      
        Zn=Z[:,n]
        #---------------------
        for m in idx_other:
            Signal = Zn-Z[:,m]
            Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)            
            #---------------------
            if detach_signal == True:
                Signal = Signal.detach()
            #---------------------
            Noise = torch.sum(Xn_grad.abs(), dim=1)*max_noise
            #---------------------
            NSR = Noise/(Signal+eps)
            NSR = NSR[NSR>alpha]
            if NSR.size(0) > 0:
                if log_NSR == True:
                    loss = loss + torch.sum(torch.log(1+NSR))
                else:
                    loss = loss + torch.sum(NSR)
                counter+=NSR.size(0)
        #---------------------           
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/(X.size(0)*(num_classes-1))
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_loss3zs_ref_max(model, X, Y, num_classes, max_noise=1, alpha=0, eps=1e-8,
                         log_NSR=False, detach_signal=False, reduction='mean'):    
    #not for single output of binary classification (len(Zn.size() <=1)
    #this loss should only be applied for correctly classified samples
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n].detach()
        if Xn.size(0) <= 0:
            continue
        idx_other=list(range(0, n))+list(range(n+1,num_classes))       
        Xn.requires_grad=True
        Z=model(Xn)      
        Zn=Z[:,n]
        Zother=Z[:,idx_other]
        Zother_max=torch.max(Zother, dim=1)[0]
        Zother_max=Zother_max.view(Zn.size())
        #---------------------
        Signal = Zn-Zother_max
        Xn_grad=torch.autograd.grad(Signal.sum(), Xn, create_graph=True)[0]
        Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)            
        #---------------------
        if detach_signal == True:
            Signal = Signal.detach()
        #---------------------
        Noise = torch.sum(Xn_grad.abs(), dim=1)*max_noise
        #---------------------
        NSR = Noise/(Signal+eps)
        NSR = NSR[NSR>alpha]
        if NSR.size(0) > 0:
            if log_NSR == True:
                loss = loss + torch.sum(torch.log(1+NSR))
            else:
                loss = loss + torch.sum(NSR)
            counter+=NSR.size(0)
        #---------------------           
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_wx_constant(model, X, Y, num_classes, mask=None, alpha=1):
    #z=w*x+b, s.t. w*x=alpha
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
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
            loss = loss + torch.sum((Signal-alpha)**2)
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_minimize_bias(model, X, Y, num_classes, mask=None, alpha=1, reduction='mean'):
    #minimize b in z=w*x+b such that |b|<=alpha
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
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
def dZdX_norm(model, X, Y, num_classes, norm):
    #z=w*x+b, return L1 norm ||w|| or L2 norm ||w||^2
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
            if norm == 1:
                loss=loss+torch.sum(Xn_grad.abs())
            elif norm == 2:
                loss=loss+torch.sum(Xn_grad**2)
            else:
                raise ValueError('unkown norm')
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_L2norm_constant(model, X, Y, num_classes, alpha=1):
    #z=w*x+b, s.t. L2 norm ||w|| = alpha
    X=X.detach()
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            loss=loss+torch.sum((torch.sum(Xn_grad**2, dim=1)-alpha**2)**2)
    loss=loss/X.size(0)
    return loss
#%%
def dZdX_L2norm_max(model, X, Y, num_classes, alpha=1, reduction='mean'):
    #z=w*x+b, s.t. L2 norm ||w|| <= alpha
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    X=X.detach()
    counter=0
    for n in range(0, num_classes):
        Xn=X[Y==n]
        if Xn.size(0) > 0:
            Xn=Xn.detach()
            Xn.requires_grad=True
            Z=model(Xn)
            if len(Z.size()) > 1:
                Zn=Z[:,n]
            else:
                Zn=Z
            Xn_grad=torch.autograd.grad(Zn.sum(), Xn, create_graph=True)[0]
            Xn_grad=Xn_grad.view(Xn_grad.size(0), -1)
            L2norm_sq = torch.sum(Xn_grad**2, dim=1)
            L2norm_sq = L2norm_sq[L2norm_sq>alpha**2]
            if L2norm_sq.size(0) >0:
                loss=loss+torch.sum(L2norm_sq)
                counter+=L2norm_sq.size(0)
    if reduction == 'counter':
        if counter>1:
            loss=loss/counter
    elif reduction == 'mean':
        loss=loss/X.size(0)
    else:
        raise ValueError('unkown reduction')
    return loss
#%%
def dZdX_jacob(model, X, Y, norm):
    #not for single output of binary classification (len(Zn.size() <=1)
    loss=torch.tensor(0.0, dtype=X.dtype, device=X.device, requires_grad=True)
    X=X.detach()
    X.requires_grad=True
    Z=model(X)
    for m in range(0, Z.size(1)):
        Zm=Z[:,m].sum()
        X_grad=torch.autograd.grad(Zm, X, create_graph=True)[0]
        if norm == 1:
            loss=loss+torch.sum(X_grad.abs())
        elif norm == 2:
            loss=loss+torch.sum(X_grad**2)
        else:
            raise ValueError('unkown norm')
    loss=loss/(X.size(0)*X.size(1)) # need to be changed to Z
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