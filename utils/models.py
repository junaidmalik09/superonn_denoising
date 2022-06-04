import torch
from torch import nn
from torch.nn import Tanh,Conv2d,Sequential,MSELoss,Module,Sigmoid


from .SelfONN import SuperONN1d, SuperONN2d, SelfONN1d, SelfONN2d
#from .SelfONN_old import SelfONNLayer,SuperONNLayer
import numpy as np

#from bm3d import bm3d_rgb,bm3d

def get_model(model_name:str):
    #if model_name=='': return 
    if model_name=='selfonn': return SelfONN
    if model_name=='cnn': return CNN
    if model_name=='bm3d': return BM3D
    if model_name=='superonn': return SuperONN
    if model_name=='superonn_reflection': return SuperONNReflection
    if model_name=='superonn_reflection_relu': return SuperONNReflectionReLU
    if model_name=='superonn_residual': return SuperONNResidual
    if model_name=='superonn_residual_normed': return SuperONNResidualNormed
    if model_name=='superonn_residual_relu': return SuperONNResidualReLU
    if model_name=='superonn_relu': return SuperONNReLU
    if model_name=='superonn_normed': return SuperONNNormed
    if model_name=='superonn_clamped': return SuperONNClamped
    if model_name=='dncnn': return DnCNN
    if model_name=='rdn': return RDN
    if model_name=='restormer': return Restormer
    if model_name=='ressuperonn': return ResSuperONN
    if model_name=='ressuperonn_clamped': return ResSuperONNClamped
    if model_name=='drso': return DeepResSuperONN
    if model_name=='drso_relu': return DeepResSuperONNReLU
    if model_name=='drso_normed': return DeepResSuperONNNormed
    if model_name=='drso_reflection_normed': return DeepResSuperONNReflectionNormed
    if model_name=='drcnn': return DeepResCNN
    if model_name=='drcnn_mix': return DeepResCNNMix

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
    def __init__(self,num_channels,num_neurons,num_layers=4,filter_size=3,q=3,max_shifts=[0,10,10],rounded_shifts=False,**kwargs):
        super(SuperONN,self).__init__()
        self.features = num_neurons//num_layers
        self.q = q
        self.model = Sequential(
            SuperONN2d(num_channels,self.features,filter_size,q=q,padding=filter_size//2,max_shift=max_shifts[0],rounded_shifts=rounded_shifts),
            Tanh(),
            *[
                m for _ in range(num_layers-2) for m in [
                    SuperONN2d(self.features,self.features,filter_size,q=q,padding=filter_size//2,max_shift=max_shifts[1],rounded_shifts=rounded_shifts),
                    Tanh(),
                ]
            ],
            SuperONN2d(self.features,num_channels,filter_size,q=q,padding=filter_size//2,max_shift=max_shifts[2],rounded_shifts=rounded_shifts),
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
            SelfONN2d(num_channels,self.features,filter_size,q=q,padding=filter_size//2),
            Tanh(),
            *[
                m for _ in range(num_layers-2) for m in [
                    SelfONN2d(self.features,self.features,filter_size,q=q,padding=filter_size//2),
                    Tanh(),
                ]
            ],
            SelfONN2d(self.features,num_channels,filter_size,q=q,padding=filter_size//2),
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
        if num_channels==1: self.model = BM3D # changed to BM3D
        else: self.model = BM3D 
    
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



class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

#DNCNN self, num_channels, num_neurons=1024, num_layers=17,filter_size=3,**kwargs

#num_channels,num_neurons,num_layers,q,max_shifts,rounded_shifts
class RDN(nn.Module):
    def __init__(self, num_channels, num_neurons, growth_rate, num_blocks, num_layers, **kwargs):
        super(RDN, self).__init__()
        self.G0 = num_neurons
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_neurons, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_neurons, num_neurons, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        """assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )"""

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        #x = self.upscale(x)
        x = self.output(x)
        return x