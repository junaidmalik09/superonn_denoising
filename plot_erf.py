import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
import glob
import utils
from utils.datasets import TestDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import cv2


df_all = pd.read_pickle('df_all_with_results.pkl')
df_dncnn = pd.read_pickle('df_dncnn.pkl')

PROJECT_NAME = 'bm3d_vs_superonn'

# load checkpoints
ckpt_path = r"{project}//{id}//checkpoints//*.ckpt"

def effective_receptive_field(model,input_data,patch_size):
    input_data.requires_grad = True
    output = model(input_data)
    custom_grad = torch.zeros_like(output)
    _,_,h,w = output.shape
    p = patch_size//2
    custom_grad.data[:,:,h//2-p : h//2+p, w//2-p : w//2+p ] = 1.
    output.backward(custom_grad)
    input_grad_abs = input_data.grad.data.abs()
    input_grad_abs_thresholded = (input_grad_abs>1e-5).float()
    input_grad_normalized = input_grad_abs.sub_(input_grad_abs.min()).div_(input_grad_abs.max()-input_grad_abs.min())
    input_data.grad.data.zero_()
    p *= 2
    return input_grad_normalized[:,:,h//2-p : h//2+p, w//2-p : w//2+p ],input_grad_abs_thresholded[:,:,h//2-p : h//2+p, w//2-p : w//2+p ]

def get_model_from_id(ID,model_name='superonn'):
    ckpt_path = r"{project}//{id}//checkpoints//*.ckpt"
    ckpt_path_now = glob.glob( ckpt_path.format(project=PROJECT_NAME,id=ID) )[0]
    model = utils.models.get_model(model_name)
    dn = utils.denoiser.Denoiser.load_from_checkpoint(ckpt_path_now,model=model,strict=False,map_location='cpu')
    return dn.model

def put_rectangle(image,start_point,end_point,color,thickness=4):
    if isinstance(image,torch.Tensor): image = image.squeeze().cpu().data.numpy()
    return cv2.rectangle(image, start_point, end_point, color, thickness)

def plot_image_on_axis(image,ax,title=None):
    if isinstance(image,torch.Tensor): image = image.squeeze().cpu().data.numpy()
    ax.imshow(image,cmap="gray")
    ax.axis('off')
    if title is not None: ax.set_title(title)

def get_cropped_image(image,start_point,end_point):
    return image[:,:,start_point[1]:end_point[1],start_point[0]:end_point[0]]

sigma = 60
num_neurons = 256
patch_size = 60

df_dncnn = df_dncnn.loc[:,['ID','q','num_neurons','sigma','max_shifts','Kodak','McM','CBSD68']].sort_values(['q','num_neurons','sigma','max_shifts'])

df_test = df_all.loc[
    (df_all.num_neurons==num_neurons) & 
    (df_all.sigma==sigma) & 
    (df_all.max_shifts.isin(['[0,0,0]','[0,5,0]']))
]

cnns = df_test.loc[(df_test.q == 1) & (df_test.max_shifts.isin(['[0,0,0]']))]
self_onns = df_test.loc[(df_test.q ==3) & (df_test.max_shifts.isin(['[0,0,0]'])) ]
super_onns = df_test.loc[(df_test.q ==3) & (df_test.max_shifts.isin(['[0,5,0]'])) ]
dncnns = df_dncnn[ df_dncnn.sigma==sigma ]

model_cnn = get_model_from_id(cnns.ID.values[0]).cuda()
model_selfonn = get_model_from_id(self_onns.ID.values[0]).cuda()
model_superonn = get_model_from_id(super_onns.ID.values[0]).cuda()




#ax[0,0].set_title('NOISY IMAGE')
#ax[0,1].set_title('CNN OUTPUT')
#ax[0,2].set_title('SELFONN OUTPUT')
#ax[0,3].set_title('SUPERONN OUTPUT')
#ax[0,4].set_title('DNCNN OUTPUT')


for dataset_name in ['CBSD68']:
    test_path = Path("datasets/test")
    ds = TestDataset(str(test_path.joinpath(dataset_name)),sigma=sigma,clip=True,num_channels=1)
    dl = DataLoader(ds,shuffle=False,pin_memory=True)

    for idx,image in enumerate(dl):
        _,_,h,w = image[0].shape
        noisy = image[0].cuda()
        clean = image[1].cuda()

        p = patch_size//2
        noisy_with_rectangle = put_rectangle(noisy.cpu(),(w//2-p, h//2-p),(w//2+p, h//2+p),(1,0,0))
        
        with torch.no_grad():
            output_cnn = model_cnn(noisy)[:,:,h//2-p : h//2+p, w//2-p : w//2+p ]
            output_selfonn = model_selfonn(noisy)[:,:,h//2-p : h//2+p, w//2-p : w//2+p ]
            output_superonn = model_superonn(noisy)[:,:,h//2-p : h//2+p, w//2-p : w//2+p ]
        torch.cuda.empty_cache()

        p = patch_size
        noisy_cropped = noisy[:,:,h//2-p : h//2+p, w//2-p : w//2+p ].cuda()
        clean_cropped = clean[:,:,h//2-p : h//2+p, w//2-p : w//2+p ].cuda()

        

        erf_cnn_norm,erf_cnn_th = effective_receptive_field(model_cnn,noisy,patch_size)
        erf_selfonn_norm,erf_selfonn_th = effective_receptive_field(model_selfonn,noisy,patch_size)
        erf_superonn_norm,erf_superonn_th = effective_receptive_field(model_superonn,noisy,patch_size)


        grid_size = (4,5)

        ax_full_noisy_image = plt.subplot2grid(grid_size, (0, 0), rowspan=3,colspan=1)
        ax_erf_cnn_norm = plt.subplot2grid(grid_size, (0, 1), rowspan=1,colspan=1)
        ax_erf_selfonn_norm = plt.subplot2grid(grid_size, (0, 2), rowspan=1,colspan=1)
        ax_erf_superonn_norm = plt.subplot2grid(grid_size, (0, 3), rowspan=1,colspan=1)
        ax_erf_cnn_th = plt.subplot2grid(grid_size, (1, 1), rowspan=1,colspan=1)
        ax_erf_selfonn_th = plt.subplot2grid(grid_size, (1, 2), rowspan=1,colspan=1)
        ax_erf_superonn_th = plt.subplot2grid(grid_size, (1, 3), rowspan=1,colspan=1)
        ax_erf_cnn_mask = plt.subplot2grid(grid_size, (2, 1), rowspan=1,colspan=1)
        ax_erf_selfonn_mask = plt.subplot2grid(grid_size, (2, 2), rowspan=1,colspan=1)
        ax_erf_superonn_mask = plt.subplot2grid(grid_size, (2, 3), rowspan=1,colspan=1)
        ax_erf_cnn_denoised = plt.subplot2grid(grid_size, (3, 1), rowspan=1,colspan=1)
        ax_erf_selfonn_denoised = plt.subplot2grid(grid_size, (3, 2), rowspan=1,colspan=1)
        ax_erf_superonn_denoised = plt.subplot2grid(grid_size, (3, 3), rowspan=1,colspan=1)

        plot_image_on_axis(noisy_with_rectangle,ax_full_noisy_image)
        plot_image_on_axis(erf_cnn_norm,ax_erf_cnn_norm)
        plot_image_on_axis(erf_selfonn_norm,ax_erf_selfonn_norm)
        plot_image_on_axis(erf_superonn_norm,ax_erf_superonn_norm)
        plot_image_on_axis(erf_cnn_th,ax_erf_cnn_th)
        plot_image_on_axis(erf_selfonn_th,ax_erf_selfonn_th)
        plot_image_on_axis(erf_superonn_th,ax_erf_superonn_th)
        plot_image_on_axis(erf_cnn_th.mul(noisy_cropped),ax_erf_cnn_mask)
        plot_image_on_axis(erf_selfonn_th.mul(noisy_cropped),ax_erf_selfonn_mask)
        plot_image_on_axis(erf_superonn_th.mul(noisy_cropped),ax_erf_superonn_mask)
        plot_image_on_axis(output_cnn,ax_erf_cnn_denoised)
        plot_image_on_axis(output_selfonn,ax_erf_selfonn_denoised)
        plot_image_on_axis(output_superonn,ax_erf_superonn_denoised)

        #plt.tight_layout()
        #plt.show()
        plt.savefig('erf_synthetic/'+dataset_name+'_'+str(idx)+'.png')