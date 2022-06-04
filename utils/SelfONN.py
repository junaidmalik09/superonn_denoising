import torch
import math
from torch import nn
import numpy as np
from pathlib import Path
from math import factorial
import matplotlib.pyplot as plt
import kornia

def randomshift(x,shifts): return kornia.geometry.transform.translate(x,shifts.to(x.device),align_corners=False)

class SuperONNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        q: int, 
        bias: bool = True,
        padding: int = 0,
        dilation: int = 1,
        learnable: bool = False,
        max_shift: float = 0,
        rounded_shifts: bool = False
    ) -> None:
        super(SuperONNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.q = q
        self.learnable = learnable
        self.max_shift = max_shift
        self.padding = padding
        self.dilation = dilation
        self.rounded_shifts = rounded_shifts
        
        self.weights = nn.Parameter(torch.Tensor(self.out_channels,self.q*self.in_channels,self.kernel_size,self.kernel_size)) # Q x C x K x D
        
        if bias: self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else: self.register_parameter('bias', None)
        
        
        if self.learnable:
            self.shifts_x = nn.Parameter(torch.Tensor(self.in_channels,1))
            self.shifts_y = nn.Parameter(torch.Tensor(self.in_channels,1))

        else:
            self.register_buffer('shifts_x',torch.Tensor(self.in_channels,1))
            self.register_buffer('shifts_y',torch.Tensor(self.in_channels,1))
        
        self.reset_parameters()
        print("SuperONNLayer initialized with shifts:",max_shift,self.rounded_shifts)

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.shifts_x,-self.max_shift,self.max_shift)
        nn.init.uniform_(self.shifts_y,-self.max_shift,self.max_shift)
        if self.rounded_shifts:
            with torch.no_grad():
                self.shifts_x.data.round_()
                self.shifts_y.data.round_()
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights,gain=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        #print(self.shifts_x,self.shifts_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1,0,2,3)
        x = randomshift(x,torch.cat([self.shifts_x,self.shifts_y],dim=1))
        x = x.permute(1,0,2,3)
        x = torch.cat([(x**i) for i in range(1,self.q+1)],dim=1)
        x = torch.nn.functional.conv2d(x,self.weights,bias=self.bias,padding=self.padding,dilation=self.dilation)        
        return x

    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', q={q}')
        if self.bias is None: s += ', bias=False'
        return s.format(**self.__dict__)


class SelfONNLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,q=1,sampling_factor=1,idx=-1,dir=[],debug=False,output=False,vis=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.q = q
        self.sampling_factor = sampling_factor
        self.weights = nn.Parameter(torch.Tensor(out_channels,q*in_channels,kernel_size,kernel_size)) # Q x C x K x D
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.q = q
        self.dir = dir
        self.debug = debug
        self.idx = idx #deprecated
        self.output = output #deprecated
        self.reset_parameters_like_torch()
        
                
    def reset_parameters(self):
        bound = 0.01
        nn.init.uniform_(self.bias,-bound,bound)
        for q in range(self.q): nn.init.xavier_uniform_(self.weights[q])
        
    def reset_parameters_like_torch(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights,gain=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        # Input to layer
        x = torch.cat([(x**i) for i in range(1,self.q+1)],dim=1)
        x = torch.nn.functional.conv2d(x,self.weights,bias=self.bias,padding=self.padding,dilation=self.dilation)        
        return x
