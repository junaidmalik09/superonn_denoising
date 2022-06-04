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


def get_model_from_id(ID,model_name='superonn'):
    ckpt_path = r"{project}//{id}//checkpoints//*.ckpt"
    ckpt_path_now = glob.glob( ckpt_path.format(project=PROJECT_NAME,id=ID) )[0]
    model = utils.models.get_model(model_name)
    dn = utils.denoiser.Denoiser.load_from_checkpoint(ckpt_path_now,model=model,strict=False,map_location='cpu')
    return dn.model

def put_rectangle(image,start_point,end_point,color,thickness=4):
    if isinstance(image,torch.Tensor): image = image.squeeze().cpu().data.numpy()
    return cv2.rectangle(image, start_point, end_point, color, thickness)

def plot_image_on_axis(image,ax):
    if isinstance(image,torch.Tensor): image = image.squeeze().cpu().data.numpy()
    ax.imshow(image,cmap="gray")
    ax.axis('off')

def get_cropped_image(image,start_point,end_point):
    return image[:,:,start_point[1]:end_point[1],start_point[0]:end_point[0]]

sigma = 60
num_neurons = 256

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
model_dncnn = get_model_from_id(dncnns.ID.values[0],model_name='dncnn').cuda()


test_path = Path("datasets/test")
ds = TestDataset(str(test_path.joinpath('McM')),sigma=sigma,clip=True,num_channels=1)
dl = DataLoader(ds,shuffle=True,pin_memory=True)

for idx,image in enumerate(dl):
        
    noisy = image[0][:,:,:512,:512].cuda()
    output_cnn = model_cnn(noisy)
    output_selfonn = model_selfonn(noisy)
    output_superonn = model_superonn(noisy)
    output_dncnn = model_dncnn(noisy)

    size = 128
    start_point = (np.random.randint(1,512-size),np.random.randint(1,512-size))
    end_point = (start_point[0]+size,start_point[1]+size)

    cm = 1/2.54  # centimeters in inches
    fig,ax = plt.subplots(1,5,figsize=(16,4))
    plt.rcParams.update({'font.size': 26})

    #ax[0,0].set_title('NOISY IMAGE')
    #ax[0,1].set_title('CNN OUTPUT')
    #ax[0,2].set_title('SELFONN OUTPUT')
    #ax[0,3].set_title('SUPERONN OUTPUT')
    #ax[0,4].set_title('DNCNN OUTPUT')

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


    plt.savefig('synthetic_denoising_output/'+str(idx)+'_McM.png',bbox_inches='tight',pad_inches=0)
    plt.close()