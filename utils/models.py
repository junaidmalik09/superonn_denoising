import torch
from torch import nn
from torch.nn import Tanh,Conv2d,Sequential,MSELoss,Module,Sigmoid
from .SelfONN import SelfONNLayer,SuperONNLayer
import numpy as np
from bm3d import bm3d_rgb,bm3d

def get_model(model_name:str):
    if model_name=='selfonn': return SelfONN
    if model_name=='cnn': return CNN
    if model_name=='bm3d': return BM3D
    if model_name=='superonn': return SuperONN
    if model_name=='dncnn': return DnCNN

class DnCNN(nn.Module):
    def __init__(self, num_channels, num_neurons=1024, num_layers=17,filter_size=3,**kwargs):
        super(DnCNN, self).__init__()
        print('DnCNN initialized..')
        padding = 1
        features = num_neurons//16
        layers = []
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=features, kernel_size=filter_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=filter_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=num_channels, kernel_size=filter_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

class SuperONN(Module):
    def __init__(self,num_channels,num_neurons,num_layers=4,filter_size=3,q=3,max_shifts=[0,10,10],rounded_shifts=False):
        super(SuperONN,self).__init__()
        self.features = num_neurons//num_layers
        self.q = q
        self.model = Sequential(
            SuperONNLayer(num_channels,self.features,filter_size,q=q,padding=filter_size//2,max_shift=max_shifts[0],rounded_shifts=rounded_shifts),
            Tanh(),
            *[
                m for _ in range(num_layers-2) for m in [
                    SuperONNLayer(self.features,self.features,filter_size,q=q,padding=filter_size//2,max_shift=max_shifts[1],rounded_shifts=rounded_shifts),
                    Tanh(),
                ]
            ],
            SuperONNLayer(self.features,num_channels,filter_size,q=q,padding=filter_size//2,max_shift=max_shifts[2],rounded_shifts=rounded_shifts),
            Sigmoid()
        )
        print("Initialized SuperONN with {0} neurons, num layers {1} filter size {2} and Q={3} and shifts rounded {4}".format(num_neurons,num_layers,filter_size,q,rounded_shifts))

    def forward(self,x):
        return self.model(x)

class SelfONN(Module):
    def __init__(self,num_channels,num_neurons,num_layers=4,filter_size=3,q=3,shift=0):
        super(SelfONN,self).__init__()
        self.features = num_neurons//num_layers
        self.q = q
        self.model = Sequential(
            SelfONNLayer(num_channels,self.features,filter_size,q=q,padding=filter_size//2),
            Tanh(),
            *[
                m for _ in range(num_layers-2) for m in [
                    SelfONNLayer(self.features,self.features,filter_size,q=q,padding=filter_size//2),
                    Tanh(),
                ]
            ],
            SelfONNLayer(self.features,num_channels,filter_size,q=q,padding=filter_size//2),
            Sigmoid()
        )
        print("Initialized Self-ONN with {0} neurons, num layers {1} filter size {2} and Q={3}".format(num_neurons,num_layers,filter_size,q))

    def forward(self,x):
        return self.model(x)

class CNN(Module):
    def __init__(self,num_channels,num_neurons,filter_size=3):
        super(CNN,self).__init__()
        self.features = num_neurons//2
        self.model = Sequential(
            Conv2d(num_channels,self.features,filter_size,padding=filter_size//2),
            Tanh(),
            Conv2d(self.features,self.features,filter_size,padding=filter_size//2),
            Tanh(),
            Conv2d(self.features,num_channels,filter_size,padding=filter_size//2),
            Tanh()
        )
        print("Initialized CNN with {0} neurons and filter size {1}".format(num_neurons,filter_size))

    def forward(self,x):
        return self.model(x)


class BM3D(Module):
    def __init__(self,num_channels,sigma):
        super().__init__()
        self.sigma = sigma/255.
        self.num_channels = num_channels
        if num_channels==1: self.model = bm3d
        else: self.model = bm3d_rgb
    
    def forward(self,x):
        _,c,h,w = x.shape
        device = x.device
        y = []
        for x_now in x:
            x_np = x_now.permute(1,2,0).cpu().numpy().squeeze()  # reshape to channels last
            y_est = self.model(x_np,self.sigma) # denoise
            y_est = torch.from_numpy(y_est).view(h,w,c).permute(2,0,1).to(x.device) # reshape to channels first
            y.append(y_est)
        return torch.stack(y,dim=0).to(device)

        