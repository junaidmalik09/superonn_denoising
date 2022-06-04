import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
import glob
import utils
from utils.datasets import TestDataset
from torch.utils.data import DataLoader

PROJECT_NAME = 'bm3d_vs_superonn'
FILENAME = 'df_dncnn.pkl'

df = pd.read_pickle(FILENAME)

# test_path
test_path = Path("datasets/test")
test_datasets = ['Kodak','McM','CBSD68']

for test_ds in test_datasets: df[test_ds] = 0

# load checkpoints
ckpt_path = r"{project}//{id}//checkpoints//*.ckpt"

for index,row in df.iterrows():
    # format path for checkpoint
    ckpt_path_now = glob.glob( ckpt_path.format(project=PROJECT_NAME,id=row.ID) )[0]
    
    # load checkpoint
    model = utils.models.get_model('dncnn')
    dn = utils.denoiser.Denoiser.load_from_checkpoint(ckpt_path_now,model=model,strict=False,map_location='cpu')
    trainer = pl.Trainer(gpus=1,weights_summary=None,progress_bar_refresh_rate=0)

    # load datasets and evaluate
    for test_dataset in test_datasets:
        
        print(test_dataset)    
        ds = TestDataset(str(test_path.joinpath(test_dataset)),sigma=row['sigma'],clip=row['clip'],num_channels=row['num_channels'])
        dl = DataLoader(ds,pin_memory=True,shuffle=False)
        res=trainer.test(model=dn,test_dataloaders=[dl],verbose=False)
        
        print(res)

        df.loc[df['ID']==row['ID'],test_dataset] = res[0]['test_psnr']
    
df.to_pickle('df_dncnn.pkl')