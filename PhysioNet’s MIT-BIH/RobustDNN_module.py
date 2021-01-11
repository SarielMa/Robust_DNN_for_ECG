import torch
import torch.nn as nn
import torch.nn.functional as nnF
import math
#%%
def Sparsify_mask(x, percent):
    #x: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
    with torch.no_grad():
        Mask=torch.ones(x.size(), dtype=x.dtype, device=x.device)
        idx_cut=int(percent*x.size(1))
        Mask[:,0:idx_cut]=0
        _, idx_sort=x.sort(dim=1, descending=False)
        idx_x = idx_sort.argsort(dim=1)
        Mask=torch.gather(Mask, 1, idx_x)
    #
    return Mask
#%%
def Sparsify(x, percent):
    m = Sparsify_mask(x, percent)
    y = m*x
    return y
#%%
class Gate(nn.Module):
    def __init__(self, num_features=None, bias=False, activation = 'relu'):
        super().__init__()
        if bias == True:
            if num_features is None:
                raise ValueError('num_features should not be None')
            self.bias = nn.Parameter(torch.zeros(num_features))
        elif bias == False:
            self.bias = None
        else:
            raise ValueError('bias should be True or False.')
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError('activation should be relu or softplus or None.')
    def forward(self, x, s=None):
        if self.activation is None:
            if self.bias is None:
                return x
            else:
                x_size = x.size()
                x = x.view(x_size[0], x_size[1], -1)
                b = self.bias.view(1,-1,1)
                y = x - b
                y = y.view(x_size)
                return y
        #-------------------------------
        #x, s: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        #-------------------------------
        #if s is None:
        #    s = x.detach()
        #y = x*self.activation(torch.tanh(s-b))
        #-------------------------------
        if self.bias is not None:
            x_size = x.size()
            x = x.view(x_size[0], x_size[1], -1)
            b = self.bias.view(1,-1,1)
            m = (x>b).to(x.dtype)
            y = self.activation(x-b) + m*b.detach()
            y = y.view(x_size)
        else:
            y = self.activation(x)
        return y
#%%
class DoubleGate(nn.Module):
    def __init__(self, in_features=None, bias=False, inplace=False):
        super().__init__()
        if bias == True:
            if in_features is None:
                raise ValueError('in_features should not be None')
            self.bias = nn.Parameter(torch.zeros(in_features, dtype=torch.float32))
        elif bias == False:
            self.bias = None
        else:
            raise ValueError('bias should be True or False')
        self.activation = nn.ReLU(inplace)
    def forward(self, x, s=None):
        #x, s: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        #-------------------------------
        #if s is None:
        #    s = x.detach()
        #y1 = x*self.activation(torch.tanh(s-b))
        #y2 = x*self.activation(torch.tanh(-s-b))
        #-------------------------------
        if self.bias is not None:
            x_size = x.size()
            x = x.view(x_size[0], x_size[1], -1)
            b = self.bias.view(1,-1,1)
            m1 = (x>b).to(x.dtype)
            y1 = self.activation(x-b) + m1*b.detach()
            y1 = y1.view(x_size)
            m2 = (x<-b).to(x.dtype)
            y2 = self.activation(-x-b) + m2*b.detach()
            y2 = y2.view(x_size)
        else:
            y1 = self.activation(x)
            y2 = self.activation(-x)
        y = torch.cat([y1, y2], dim=1)
        return y
#%%
class Normalization(nn.Module):
    def __init__(self, layerA, layerB, num_features=None,
                 eps=1e-5, gain_min=1.0, momentum=0.1, track_running_stats=True):
        # layerA -> relu -> Normalization -> layerB
        super().__init__()
        self.layerA=layerA
        self.layerB=layerB
        if layerA is not None:
            num_features = layerA.weight.size(0)
        elif layerB is not None:
            num_features = layerB.weight.size(1)
        self.register_buffer('eps', torch.tensor(eps, dtype=torch.float32))
        self.register_buffer('gain_min', torch.tensor(gain_min, dtype=torch.float32))
        self.register_buffer('momentum', torch.tensor(momentum, dtype=torch.float32))
        self.register_buffer('gain', torch.ones(num_features, dtype=torch.float32))
        self.track_running_stats=track_running_stats

    def set_momentum(self, momentum):
        self.momentum.fill_(momentum)

    def update_gain(self, x):
        #x: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        with torch.no_grad():
            x = x.view(x.size(0), x.size(1), -1)
            mask = torch.zeros_like(x)
            mask[x>self.eps]=1
            counter = torch.sum(mask, dim=(0,2))
            if torch.sum(counter).item() > 0:
                batch_mean = torch.sum(x*mask, dim=(0,2))/(counter+1e-16)
                #print('batch_mean', batch_mean)
                #print('counter', counter)
                temp = counter > 0
                #print(temp)
                self.gain[temp] *= 1 - self.momentum
                self.gain[temp] += self.momentum*batch_mean[temp]
                self.gain[self.gain<self.gain_min]=self.gain_min

    def update_A(self, x):
        #x: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        with torch.no_grad():
            w = self.layerA.weight.data
            w = w.view(w.size(0), -1)
            alpha = torch.sqrt(torch.sum(w**2, dim=1, keepdim=True))+1e-16
            w /= alpha
            #print(alpha)
            alpha = alpha.squeeze()
            alpha = alpha.view(1, -1, 1)
            return alpha

    def update_B(self, g):
        with torch.no_grad():
            w = self.layerB.weight.data
            w = w.view(w.size(0), w.size(1), -1)
            w *= g

    def update_parameter(self, x):
        #x: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        if self.layerA is not None:
            alpha = self.update_A(x)
            x = x*alpha
        else:
            alpha=1
        #---------------------------------------------------
        gain_old = self.gain.clone().view(1,-1,1)
        self.update_gain(x)
        gain_new = self.gain.view(1,-1,1)
        #---------------------------------------------------
        if self.layerB is not None:
            if alpha is not None:
                g = (gain_new/gain_old)/alpha
            else:
                g = gain_new/gain_old
            self.update_B(g)
        #---------------------------------------------------
        return x.view(x_size)

    def forward(self, x):
        if self.training == True and self.track_running_stats == True:
            x = self.update_parameter(x)
        #---------------------------------------
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        x = x/self.gain.view(1,-1,1)
        x = x.view(x_size)
        return x
#%%
class SimpleNormalization(nn.Module):
    def __init__(self, layerA, layerB, num_features=None,
                 eps=1e-5, gain_min=1.0, momentum=0.1, track_running_stats=True):
        # layerA -> relu -> Normalization -> layerB
        super().__init__()
        self.layerA=layerA
        self.layerB=layerB
        self.register_buffer('eps', torch.tensor(eps, dtype=torch.float32))
        self.register_buffer('gain_min', torch.tensor(gain_min, dtype=torch.float32))
        self.register_buffer('momentum', torch.tensor(momentum, dtype=torch.float32))
        self.register_buffer('gain', torch.ones(1, dtype=torch.float32))
        self.track_running_stats=track_running_stats

    def set_momentum(self, momentum):
        self.momentum.fill_(momentum)
        
    def update_gain(self, x):
        #x: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        with torch.no_grad():
            x = x.view(x.size(0), x.size(1), -1)
            mask = torch.zeros_like(x)
            mask[x>self.eps]=1
            counter = torch.sum(mask)
            if counter > 0:
                batch_sum = torch.sum(x*mask)
                batch_mean = batch_sum/counter
                self.gain *= 1 - self.momentum
                self.gain += self.momentum*batch_mean
                self.gain[self.gain<self.gain_min]=self.gain_min

    def update_A(self, x):
        #x: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        with torch.no_grad():
            w = self.layerA.weight.data
            w = w.view(w.size(0), -1)
            alpha = torch.sqrt(torch.sum(w**2)/w.size(0))
            w /= alpha
            return alpha

    def update_B(self, g):
        with torch.no_grad():
            self.layerB.weight.data *= g

    def update_parameter(self, x):
        #x: (N, L) or (N, C, L) or (N, C, H, W) or (N, C, D, H, W)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        if self.layerA is not None:
            alpha = self.update_A(x)
            x = x*alpha
        else:
            alpha=1
        #---------------------------------------------------
        gain_old = self.gain.clone()
        self.update_gain(x)
        gain_new = self.gain
        #---------------------------------------------------
        if self.layerB is not None:
            if alpha is not None:
                g = (gain_new/gain_old)/alpha
            else:
                g = gain_new/gain_old
            self.update_B(g)
        #---------------------------------------------------
        return x.view(x_size)

    def forward(self, x):
        if self.training == True and self.track_running_stats == True:
            x = self.update_parameter(x)
        #---------------------------------------
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], -1)
        x = x/self.gain
        x = x.view(x_size)
        return x
#%%
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self.register_buffer('WoW', torch.zeros(self.weight.size(), dtype=torch.float32))
        self.register_buffer('W_mask', torch.ones(self.weight.size(), dtype=torch.float32))
        self.normalize_kernel()
        if self.bias is not None:
            self.bias.data.fill_(0)

    def disable_bias(self):
        if self.bias is not None:
            self.bias.data.fill_(0)
            self.bias.requires_grad=False

    def find_dead_kernel(self):
        WoW = self.WoW # WoW should be abs(dLdW)
        indexlist=[]
        for n in range(0, WoW.size(0)):
            if torch.sum(WoW[n]).item() == 0:
                indexlist.append(n)
        return indexlist

    def initialize_dead_kernel(self, with_bias=True):
        indexlist = self.find_dead_kernel()
        with torch.no_grad():
            w = self.weight
            b = self.bias
            for idx in indexlist:
                v = torch.randn_like(w[idx])
                g = torch.sqrt(torch.sum(v**2))+1e-16
                w[idx] = v/g
                if with_bias == True and b is not None:
                    b[idx] = 0
        return len(indexlist)

    def zero_dead_kernel(self, with_bias=True):
        with torch.no_grad():
            WoW = self.WoW
            b = self.bias
            for n in range(0, WoW.size(0)):
                if torch.sum(WoW[n]).item() == 0:
                    self.W_mask[n]=0
                    self.weight[n]=0
                    if with_bias == True and b is not None:
                        b[n] = 0

    def normalize_kernel(self, alpha=1):
        #set alpha=2**0.5 for relu
        with torch.no_grad():
            w = self.weight
            g = alpha*torch.sqrt(1/(torch.sum(w**2, dim=1, keepdim=True)+1e-16))
            w *= g
            if self.bias is not None:
                self.bias *= g.squeeze()

    def normalize_weight(self, alpha=1):
        #set alpha=2**0.5 for relu
        with torch.no_grad():
            w = self.weight
            g = alpha*torch.sqrt(w.size(0)/torch.sum(w**2))
            w *= g
            if self.bias is not None:
                self.bias *= g

    def zero_WoW(self):
        self.WoW.fill_(0)

    def update_WoW(self, ):
        self.WoW+=self.weight.grad.abs()

    def normalize_WoW(self):
        WoW = self.WoW
        WoW_max = WoW.max(dim=1, keepdim=True)[0]
        WoW /= WoW_max+1e-16

    def truncate_weight(self, percent=None, threshold=None):
        with torch.no_grad():
            w = self.weight.view(-1)
            w_abs = w.abs()
            if percent is not None and threshold is None:
                 idx_cut=int(percent*w.size(0))
                 _, idx_sort=w_abs.sort(descending=False)
                 w[idx_sort[0:idx_cut]]=0
                 #-----------------------
                 m = self.W_mask.view(-1)
                 m.fill_(1)
                 m[idx_sort[0:idx_cut]]=0
            elif percent is None and threshold is not None:
                temp = w_abs<threshold
                self.linear.weight[temp]=0
                self.W_mask.fill_(1)
                self.W_mask[temp]=0

    def truncate_weight_by_WoW(self, percent=None, threshold=None):
        with torch.no_grad():
            if percent is not None:
                 w = self.weight.view(-1)
                 WoW = self.WoW.view(-1)
                 idx_cut=int(percent*w.size(0))
                 _, idx_sort=WoW.sort(descending=False)
                 w[idx_sort[0:idx_cut]]=0
                 #-----------------------
                 m = self.W_mask.view(-1)
                 m.fill_(1)
                 m[idx_sort[0:idx_cut]]=0
            elif threshold is not None:
                temp = self.WoW<threshold
                self.weight[temp]=0
                self.W_mask.fill_(1)
                self.W_mask[temp]=0

    def truncate_kernel(self, percent):
        with torch.no_grad():
            for n in range(0, self.linear.weight.size(0)):
                w = self.weight[n].view(-1)
                w_abs = w.abs()
                idx_cut=int(percent*w.size(0))
                _, idx_sort=w_abs.sort(descending=False)
                w[idx_sort[0:idx_cut]]=0
                #-----------------------
                m = self.W_mask[n].view(-1)
                m.fill_(1)
                m[idx_sort[0:idx_cut]]=0

    def truncate_weight_by_mask(self):
        with torch.no_grad():
            self.weight[self.W_mask==0]=0

    def forward(self, x):
        x = self.linear(x)
        return x
#%%
class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation='relu', double_gate=False, batch_norm=False):
        super().__init__()
        if double_gate == True:
            if (out_features%2) != 0:
                raise ValueError('out_features should be an even number.')
            linear_out_features=int(out_features/2)
            self.gate = DoubleGate(linear_out_features, bias, activation)
        else:
            linear_out_features=out_features
            self.gate = Gate(out_features, bias, activation)
        if activation is None and bias == False:
            self.gate = nn.Identity()

        self.linear = Linear(in_features, linear_out_features, bias=False)

        if batch_norm == True and activation is not None:
            self.bn = Normalization(None, None, out_features)
        else:
            self.bn = nn.Identity()

    def set_Normalization(self, freeze, momentum):
        if not isinstance(self.bn, nn.Identity):
            self.bn.track_running_stats = not freeze
            self.bn.set_momentum(momentum)

    def forward(self, x):
        x = self.linear(x)
        x = self.gate(x)
        x = self.bn(x)
        return x
#%%
class ConvNd(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.dim=dim
        if dim == 1:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, groups,
                                  bias=bias, padding_mode=padding_mode)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, groups,
                                  bias=bias, padding_mode=padding_mode)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, groups,
                                  bias=bias, padding_mode=padding_mode)
        self.weight=self.conv.weight
        self.bias=self.conv.bias
        self.register_buffer('WoW', torch.zeros(self.weight.size(), dtype=torch.float32))
        self.register_buffer('W_mask', torch.ones(self.weight.size(), dtype=torch.float32))
        self.normalize_kernel()
        if self.bias is not None:
            self.bias.data.fill_(0)
            
    def disable_bias(self):
        if self.bias is not None:
            self.bias.data.fill_(0)
            self.bias.requires_grad=False
        
    def find_dead_kernel(self):
        WoW = self.WoW # WoW should be abs(dLdW)
        indexlist=[]
        for n in range(0, WoW.size(0)):
            if torch.sum(WoW[n]).item() == 0:
                indexlist.append(n)
        return indexlist

    def initialize_dead_kernel(self, with_bias=True):
        indexlist = self.find_dead_kernel()
        b = self.bias
        with torch.no_grad():
            w = self.weight
            for idx in indexlist:
                v = torch.randn_like(w[idx])
                g = torch.sqrt(torch.sum(v**2))+1e-16
                w[idx] = v/g
                if with_bias == True and b is not None:
                    b[idx] = 0
        return len(indexlist)

    def zero_dead_kernel(self, with_bias=True):
        with torch.no_grad():
            WoW = self.WoW
            b=self.bias
            for n in range(0, WoW.size(0)):
                if torch.sum(WoW[n]).item() == 0:
                    self.W_mask[n]=0
                    self.weight[n]=0
                    if with_bias == True and b is not None:
                        b[n] = 0

    def normalize_kernel(self, alpha=1):
        #set alpha=2**0.5 for relu
        with torch.no_grad():
            w = self.weight
            w = w.view(w.size(0),-1)
            g = alpha*torch.sqrt(1/(torch.sum(w**2, dim=1, keepdim=True)+1e-16))
            w *= g
            if self.bias is not None:
                self.bias *= g.squeeze()

    def normalize_weight(self, alpha=1):
        #set alpha=2**0.5 for relu
        with torch.no_grad():
            w = self.weight
            g = alpha*torch.sqrt(w.size(0)/torch.sum(w**2))
            w *= g
            if self.bias is not None:
                self.bias *= g

    def zero_WoW(self):
        self.WoW.fill_(0)

    def update_WoW(self, ):
        self.WoW += self.weight.grad.abs()

    def normalize_WoW(self):
        w = self.WoW
        w = w.view(w.size(0),-1)
        w_max = w.max(dim=1, keepdim=True)[0]
        w /= w_max+1e-16

    def truncate_weight(self, percent=None, threshold=None):
        with torch.no_grad():
            w = self.weight.view(-1)
            w_abs = w.abs()
            if percent is not None and threshold is None:
                 idx_cut=int(percent*w.size(0))
                 _, idx_sort=w_abs.sort(descending=False)
                 w[idx_sort[0:idx_cut]]=0
                 #-----------------------
                 m = self.W_mask.view(-1)
                 m.fill_(1)
                 m[idx_sort[0:idx_cut]]=0
            elif percent is None and threshold is not None:
                temp = w_abs<threshold
                self.weight[temp]=0
                self.W_mask.fill_(1)
                self.W_mask[temp]=0

    def truncate_weight_by_WoW(self, percent=None, threshold=None):
        with torch.no_grad():
            if percent is not None:
                 w = self.weight.view(-1)
                 WoW = self.WoW.view(-1)
                 idx_cut=int(percent*w.size(0))
                 _, idx_sort=WoW.sort(descending=False)
                 w[idx_sort[0:idx_cut]]=0
                 #-----------------------
                 m = self.W_mask.view(-1)
                 m.fill_(1)
                 m[idx_sort[0:idx_cut]]=0
            elif threshold is not None:
                temp = self.WoW<threshold
                self.weight[temp]=0
                self.W_mask.fill_(1)
                self.W_mask[temp]=0

    def truncate_kernel(self, percent):
        with torch.no_grad():
            for n in range(0, self.weight.size(0)):
                w = self.weight[n].view(-1)
                w_abs = w.abs()
                idx_cut=int(percent*w.size(0))
                _, idx_sort=w_abs.sort(descending=False)
                w[idx_sort[0:idx_cut]]=0
                #-----------------------
                m = self.W_mask[n].view(-1)
                m.fill_(1)
                m[idx_sort[0:idx_cut]]=0

    def truncate_weight_by_mask(self):
        with torch.no_grad():
            self.weight[self.W_mask==0]=0

    def forward(self, x):
        x = self.conv(x)
        return x
#%%
class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(1, in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode)
#%%
class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(2, in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode)
#%%
class Conv3d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(3, in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias, padding_mode)
#%%
class Convolution(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 bias=True, activation='relu', double_gate=False, batch_norm=False):
        super().__init__()

        if double_gate == True:
            if (out_channels%2) != 0:
                raise ValueError('out_features should be an even number.')
            conv_out_channels=int(out_channels/2)
            self.gate = DoubleGate(conv_out_channels, bias, activation)
        else:
            conv_out_channels=out_channels
            self.gate = Gate(out_channels, bias, activation)
        if activation is None and bias == False:
            self.gate = nn.Identity()

        self.dim=dim
        self.conv = ConvNd(dim, in_channels, conv_out_channels, kernel_size,
                           stride, padding, dilation, groups,
                           bias=False, padding_mode=padding_mode)

        self.register_buffer('WoW', torch.zeros(self.conv.weight.size()))
        self.register_buffer('W_mask', torch.ones(self.conv.weight.size()))
        if batch_norm == True and activation is not None:
            self.bn = Normalization(None, None, out_channels)
        else:
            self.bn = nn.Identity()

    def set_Normalization(self, freeze, momentum):
        if not isinstance(self.bn, nn.Identity):
            self.bn.track_running_stats = not freeze
            self.bn.set_momentum(momentum)

    def forward(self, x):
        x = self.conv(x)
        x = self.gate(x)
        x = self.bn(x)
        return x
#%%
class ConvTransposeNd(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros',
                 anti_artifact=False):
        super().__init__()
        self.dim=dim
        if dim == 1:
            self.tconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                            stride, padding, output_padding,
                                            groups, bias, dilation, padding_mode)
        elif dim == 2:
            self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                            stride, padding, output_padding,
                                            groups, bias, dilation, padding_mode)
        elif dim == 3:
            self.tconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                            stride, padding, output_padding,
                                            groups, bias, dilation, padding_mode)
        self.anti_artifact=anti_artifact
        if anti_artifact == True:
            if dim == 1:
                self._tconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                                 stride, padding, output_padding,
                                                 groups, bias, dilation, padding_mode)
            elif dim == 2:
                self._tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride, padding, output_padding,
                                                 groups, bias, dilation, padding_mode)
            elif dim == 3:
                self._tconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride, padding, output_padding,
                                                 groups, bias, dilation, padding_mode)
            self._tconv.weight.data.fill_(1)
        #-------------------------------------------
        self.weight=self.tconv.weight
        self.bias=self.tconv.bias
        self.register_buffer('WoW', torch.zeros(self.weight.size(), dtype=torch.float32))
        self.register_buffer('W_mask', torch.ones(self.weight.size(), dtype=torch.float32))
        self.normalize_kernel()
        if self.bias is not None:
            self.bias.data.fill_(0)

    def disable_bias(self):
        if self.bias is not None:
            self.bias.data.fill_(0)
            self.bias.requires_grad=False
            
    def find_dead_kernel(self):
        WoW = self.WoW # WoW should be abs(dLdW)
        indexlist=[]
        for n in range(0, WoW.size(0)):
            if torch.sum(WoW[n]).item() == 0:
                indexlist.append(n)
        return indexlist

    def initialize_dead_kernel(self, with_bias=True):
        indexlist = self.find_dead_kernel()
        b=self.bias
        with torch.no_grad():
            w = self.weight
            for idx in indexlist:
                v = torch.randn_like(w[idx])
                g = torch.sqrt(torch.sum(v**2))+1e-16
                w[idx] = v/g
                if with_bias == True and b is not None:
                    b[idx] = 0
        return len(indexlist)

    def zero_dead_kernel(self, with_bias=True):
        with torch.no_grad():
            WoW = self.WoW
            b=self.bias
            for n in range(0, WoW.size(0)):
                if torch.sum(WoW[n]).item() == 0:
                    self.W_mask[n]=0
                    self.weight[n]=0
                    if with_bias == True and b is not None:
                        b[n] = 0

    def normalize_kernel(self, alpha=1):
        #set alpha=2**0.5 for relu
        with torch.no_grad():
            w = self.weight
            w = w.view(w.size(0),-1)
            g = alpha*torch.sqrt(1/(torch.sum(w**2, dim=1, keepdim=True)+1e-16))
            w *= g
            if self.bias is not None:
                self.bias *= g.squeeze()

    def normalize_weight(self, alpha=1):
        #set alpha=2**0.5 for relu
        with torch.no_grad():
            w = self.weight
            g = alpha*torch.sqrt(w.size(0)/torch.sum(w**2))
            w *= g
            if self.bias is not None:
                self.bias *= g

    def zero_WoW(self):
        self.WoW.fill_(0)

    def update_WoW(self, ):
        self.WoW += self.weight.grad.abs()

    def normalize_WoW(self):
        w = self.WoW
        w = w.view(w.size(0),-1)
        w_max = w.max(dim=1, keepdim=True)[0]
        w /= w_max+1e-16

    def truncate_weight(self, percent=None, threshold=None):
        with torch.no_grad():
            w = self.weight.view(-1)
            w_abs = w.abs()
            if percent is not None and threshold is None:
                 idx_cut=int(percent*w.size(0))
                 _, idx_sort=w_abs.sort(descending=False)
                 w[idx_sort[0:idx_cut]]=0
                 #-----------------------
                 m = self.W_mask.view(-1)
                 m.fill_(1)
                 m[idx_sort[0:idx_cut]]=0
            elif percent is None and threshold is not None:
                temp = w_abs<threshold
                self.weight[temp]=0
                self.W_mask.fill_(1)
                self.W_mask[temp]=0

    def truncate_weight_by_WoW(self, percent=None, threshold=None):
        with torch.no_grad():
            if percent is not None:
                 w = self.weight.view(-1)
                 WoW = self.WoW.view(-1)
                 idx_cut=int(percent*w.size(0))
                 _, idx_sort=WoW.sort(descending=False)
                 w[idx_sort[0:idx_cut]]=0
                 #-----------------------
                 m = self.W_mask.view(-1)
                 m.fill_(1)
                 m[idx_sort[0:idx_cut]]=0
            elif threshold is not None:
                temp = self.WoW<threshold
                self.weight[temp]=0
                self.W_mask.fill_(1)
                self.W_mask[temp]=0

    def truncate_kernel(self, percent):
        with torch.no_grad():
            for n in range(0, self.weight.size(0)):
                w = self.weight[n].view(-1)
                w_abs = w.abs()
                idx_cut=int(percent*w.size(0))
                _, idx_sort=w_abs.sort(descending=False)
                w[idx_sort[0:idx_cut]]=0
                #-----------------------
                m = self.W_mask[n].view(-1)
                m.fill_(1)
                m[idx_sort[0:idx_cut]]=0

    def truncate_weight_by_mask(self):
        with torch.no_grad():
            self.weight[self.W_mask==0]=0

    def forward(self, x):
        x = self.tconv(x)
        if self.anti_artifact == True:
            with torch.no_grad():
                t = torch.ones_like(x)
                t = self._tconv(t)
                t[t.abs()<1]=1
            x = x/t
        return x
#%%
class ConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros',
                 anti_artifact=False):
        super().__init__(1, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups, dilation, bias, padding_mode,
                         anti_artifact)
#%%
class ConvTranspose2d(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros',
                 anti_artifact=False):
        super().__init__(2, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups, dilation, bias, padding_mode,
                         anti_artifact)
#%%
class ConvTranspose3d(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0, groups=1, dilation=1, bias=True, padding_mode='zeros',
                 anti_artifact=False):
        super().__init__(3, in_channels, out_channels, kernel_size,
                         stride, padding, output_padding, groups, dilation, bias, padding_mode,
                         anti_artifact)
#%%
class TransposedConvolution(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding = 0,
                 dilation=1, groups=1, padding_mode='zeros',
                 bias=True, activation='relu', double_gate=False, batch_norm=False, anti_artifact=False):
        super().__init__()

        if double_gate == True:
            if (out_channels%2) != 0:
                raise ValueError('out_features should be an even number.')
            tconv_out_channels=int(out_channels/2)
            self.gate = DoubleGate(tconv_out_channels, bias, activation)
        else:
            tconv_out_channels=out_channels
            self.gate = Gate(out_channels, bias, activation)
        if activation is None and bias == False:
            self.gate = nn.Identity()

        self.dim=dim
        self.tconv = nn.ConvTransposeNd(dim, in_channels, tconv_out_channels, kernel_size,
                                        stride, padding, output_padding,
                                        groups, bias, dilation, padding_mode, anti_artifact)
        if batch_norm == True and activation is not None:
            self.bn = Normalization(None, None, out_channels)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        x = self.tconv(x)
        x = self.gate(x)
        x = self.bn(x)
        return x
#%%




