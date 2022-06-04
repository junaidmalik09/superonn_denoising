import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from utils.SelfONN import SuperONNLayer

def effective_receptive_field(model):
    input_data = torch.randn(1,1,36,36,requires_grad=True,device='cuda:0')
    output = model(input_data)
    custom_grad = torch.zeros_like(output)
    _,_,h,w = output.shape
    custom_grad.data[:,:,h//2-2 : h//2+2, w//2-2 : w//2+2 ] = 1.
    output.backward(custom_grad)
    input_grad_abs = input_data.grad.data.abs()
    input_grad_normalized = input_grad_abs #.sub_(input_grad_abs.min()).div_(input_grad_abs.max()-input_grad_abs.min())
    return input_grad_normalized

model = nn.Sequential(
    nn.Conv2d(1,16,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(16,16,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(16,16,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(16,16,3,padding=1),
    nn.ReLU(),
    nn.Conv2d(16,1,3,padding=1),
    nn.ReLU(),
).to('cuda:0')

model2 = nn.Sequential(
    SuperONNLayer(1,16,3,padding=1,q=1,max_shift=0),
    nn.Tanh(),
    SuperONNLayer(16,16,3,padding=1,q=1,max_shift=5),
    nn.Tanh(),
    SuperONNLayer(16,16,3,padding=1,q=1,max_shift=5),
    nn.Tanh(),
    SuperONNLayer(16,1,3,padding=1,q=1,max_shift=0),
    nn.Tanh(),
).to('cuda:0')


fig,ax = plt.subplots(1,2)


# cnn
erf = effective_receptive_field(model).data
print(erf.max(),erf.min())
ax[0].imshow(
    erf.squeeze().cpu().numpy(),
    cmap="gray"
)



# superonn
erf = effective_receptive_field(model2).data
print(erf.max(),erf.min())
ax[1].imshow(
    erf.squeeze().cpu().numpy(),
    cmap="gray"
)


plt.show()

