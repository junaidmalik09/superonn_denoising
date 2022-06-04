import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
import glob
import utils
from utils.datasets import SIDDValidDataset
from torch.utils.data import DataLoader,Subset
import matplotlib.pyplot as plt

import cv2

df_all = pd.read_excel('df_rw_with_results.xlsx')

PROJECT_NAME = 'bm3d_vs_superonn'

# load checkpoints
ckpt_path = r"{project}//{id}//checkpoints//*.ckpt"


def get_model_from_id(ID,model_name='superonn'):
    ckpt_path = r"{project}//{id}//checkpoints//*.ckpt"
    ckpt_path_now = glob.glob( ckpt_path.format(project=PROJECT_NAME,id=ID) )[0]
    model = utils.models.get_model(model_name)
    dn = utils.denoiser.Denoiser.load_from_checkpoint(ckpt_path_now,model=model,strict=False,map_location='cpu')
    return dn.model

def put_rectangle(image,start_point,end_point,color,thickness=4):
    if isinstance(image,torch.Tensor): image = image.squeeze().permute(1,2,0).contiguous().cpu().data.numpy()
    return cv2.rectangle(image, start_point, end_point, color, thickness)

def plot_image_on_axis(image,ax):
    if isinstance(image,torch.Tensor): image = image.squeeze().permute(1,2,0).contiguous().cpu().data.numpy()
    ax.imshow(image,cmap="gray")
    ax.axis('off')

def get_cropped_image(image,start_point,end_point):
    return image[:,:,start_point[1]:end_point[1],start_point[0]:end_point[0]]

def effective_receptive_field(model,start_point,end_point,input_data=None,patch_size=20):
    if input_data is None: input_data = torch.randn(1,3,36,36,requires_grad=True,device='cuda:0')
    else: input_data.requires_grad = True
    output = model(input_data)
    custom_grad = torch.zeros_like(output)
    _,_,h,w = output.shape
    custom_grad.data[:,:,start_point[1]:end_point[1],start_point[0]:end_point[0]] = 1.
    output.backward(custom_grad)
    input_grad_abs = input_data.mul((input_data.grad.data.abs()>0).float())
    input_grad_normalized = input_grad_abs #.sub_(input_grad_abs.min()).div_(input_grad_abs.max()-input_grad_abs.min())
    return input_grad_normalized

sigma = 60
num_neurons = 256

df_test = df_all.loc[
    (df_all.num_neurons==num_neurons) & 
    (df_all.max_shifts.isin(['[0,0,0]','[0,5,0]']))
]

cnns = df_test.loc[(df_test.q == 1) & (df_test.max_shifts.isin(['[0,0,0]']))]
self_onns = df_test.loc[(df_test.q ==3) & (df_test.max_shifts.isin(['[0,0,0]'])) ]
super_onns = df_test.loc[(df_test.q ==3) & (df_test.max_shifts.isin(['[0,5,0]'])) ]


model_cnn = get_model_from_id(cnns.ID.values[0]).cuda()
model_selfonn = get_model_from_id(self_onns.ID.values[0]).cuda()
model_superonn = get_model_from_id(super_onns.ID.values[0]).cuda()



test_path = Path("datasets/test")
ds = SIDDValidDataset()
ds = Subset(ds,torch.arange(100))
dl = DataLoader(ds,shuffle=True,pin_memory=True)

for idx,image in enumerate(dl):
        
    noisy = image[0][:,:,:256,:256].cuda()
    output_cnn = model_cnn(noisy)
    output_selfonn = model_selfonn(noisy)
    output_superonn = model_superonn(noisy)

    size = 64
    start_point = (np.random.randint(1,256-size),np.random.randint(1,256-size))
    end_point = (start_point[0]+size,start_point[1]+size)

    ref_cnn = effective_receptive_field(model_cnn,start_point,end_point,noisy)
    ref_selfonn = effective_receptive_field(model_selfonn,start_point,end_point,noisy)
    ref_superonn = effective_receptive_field(model_superonn,start_point,end_point,noisy)

    fig,ax = plt.subplots(1,5,figsize=(16,4))
    plt.rcParams.update({'font.size': 26})

    #ax[0,0].set_title('NOISY IMAGE')
    #ax[0,1].set_title('CNN OUTPUT')
    #ax[0,2].set_title('SELFONN OUTPUT')
    #ax[0,3].set_title('SUPERONN OUTPUT')


    plot_image_on_axis(put_rectangle(noisy,start_point,end_point,(1,1,1) ) ,ax[0])
    #plot_image_on_axis(put_rectangle(output_cnn,start_point,end_point,(1,1,1) ),ax[0,1])
    #plot_image_on_axis(put_rectangle(output_selfonn,start_point,end_point,(1,1,1) ),ax[0,2])
    #plot_image_on_axis(put_rectangle(output_superonn,start_point,end_point,(1,1,1) ),ax[0,3])
    #plot_image_on_axis(put_rectangle(output_dncnn,start_point,end_point,(1,1,1) ),ax[0,4])

    plot_image_on_axis(get_cropped_image(noisy,start_point,end_point),ax[1])
    plot_image_on_axis(get_cropped_image(output_cnn,start_point,end_point),ax[2])
    plot_image_on_axis(get_cropped_image(output_selfonn,start_point,end_point),ax[3])
    plot_image_on_axis(get_cropped_image(output_superonn,start_point,end_point),ax[4])
    #plot_image_on_axis(get_cropped_image(output_dncnn,start_point,end_point),ax[5])

    #plot_image_on_axis(ref_cnn,ax[2,0])
    #plot_image_on_axis(ref_cnn,ax[2,1])
    #plot_image_on_axis(ref_selfonn,ax[2,2])
    #plot_image_on_axis(ref_superonn,ax[2,3])


    plt.show()
    #plt.savefig('real_world_denoising_output/'+str(idx)+'.png',bbox_inches='tight',pad_inches=0)
    #plt.close()