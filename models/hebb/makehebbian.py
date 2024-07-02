import torch
import torch.nn as nn
from hebb import HebbianConv2d

from .base import HebbFactory

class UnsqueezeLast(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.d = d
    
    def forward(self, x):
        return x.reshape(*x.shape, *([1]*self.d))

class FlattenLast(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.d = d
    
    def forward(self, x):
        return x.reshape(*(x.shape[:-self.d-1]), -1)

def init_weights(m, init_type='normal', gain=0.02):
    if init_type == 'normal':
        nn.init.normal_(m.weight.data, 0.0, gain)
    elif init_type == 'xavier':
        nn.init.xavier_normal_(m.weight.data, gain=gain)
    elif init_type == 'kaiming':
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif init_type == 'orthogonal':
        nn.init.orthogonal_(m.weight.data, gain=gain)
    else:
        raise NotImplementedError("Unsupported initialization method {}".format(init_type))
    return m

def makehebbian(model, exclude=None, hebb_param_dict=None):
    def _match(n, e):
        #return e.endswith(n)
        return e == n
    
    hfactory = HebbFactory(hebb_param_dict)
    
    if exclude is None: exclude = []
    exclude = [(n, m) for n, m in model.named_modules() if any([_match(n, e) for e in exclude])]
    print("Layers excluded from conversion to Hebbian: {}".format([n for n, m in exclude]))
    exclude = [m for n, p in exclude for m in [*p.modules()]]
    
    def _replacelayer(m, n, l):
        #setattr(m, n, l)
        m.register_module(n, l)
    
    def _makehebbian(module):
        for n, m in module.named_children():
            if m in exclude: continue
            if type(m) is nn.Conv2d:
                if m.dilation != 1 and m.dilation != (1, 1): raise RuntimeError("Dilation not supported with Hebbian layers")
                if m.groups != 1: raise RuntimeError("Grouped convolution not supported with Hebbian layers")
                _replacelayer(module, n, init_weights(hfactory.create_hebb_layer(in_channels=m.in_channels, out_channels=m.out_channels,
                                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding, teacher_distrib=None), init_type='kaiming'))
            elif type(m) is nn.Linear:
                _replacelayer(module, n, nn.Sequential(UnsqueezeLast(2), init_weights(hfactory.create_hebb_layer(in_channels=m.in_features, out_channels=m.out_features,
                                kernel_size=1, teacher_distrib=None), init_type='kaiming'), FlattenLast(2)))
            else:
                for p in m.parameters(recurse=False): p.requires_grad = False
    
    model.apply(_makehebbian)
    
    return model

