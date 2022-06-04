import torch
from utils.denoiser import Denoiser,Denoiser_BM3D
from utils.models import get_model
from utils.datasets import TestDataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import pandas as pd
import glob
from pathlib import Path

def get_top_networks(csv_path = "wandb_export_2021-01-09T19_44_21.092+02_00.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1)
    splitted = df['Name'].str.split('_',n=1,expand=True)
    df['Run Name'] = splitted[0]
    df['Version'] = splitted[1]
    df = df.drop(columns=['Name'])
    idx = df.groupby(['Run Name'])['top_val_psnr'].transform(max) == df['top_val_psnr']
    df = df[idx]
    df = df.drop(columns=[
        'Created',
        'End Time',
        'Hostname',
        'Notes',
        'State',
        'Updated',
        'User',
        'global_step',
        'epoch'
    ])
    top_networks = df[['ID','Run Name','Version','num_channels','num_neurons','q','sigma','train_ratio','top_val_psnr','clip']]
    return top_networks
    

if __name__ == '__main__':

    torch.manual_seed(0)

    # test_path
    test_path = Path("datasets/test")
    test_datasets = ['Kodak','McM']
        
    # load trained models and evaluate dataloader
    project_name = 'bm3d_vs_selfonn'
    top_networks = get_top_networks()
    top_networks_now = top_networks.groupby(['sigma','clip']).max().reset_index()[['ID','sigma','clip','num_channels']]
    for index, row in top_networks_now.iterrows():
        # load model from checkpoint
        model = get_model('bm3d') # big messup this..
        dn = Denoiser_BM3D(row['num_channels'],row['sigma'])
        
        # load datasets and evaluate
        for test_dataset in test_datasets:    
            trainer = pl.Trainer(gpus=1)
            ds = TestDataset(str(test_path.joinpath(test_dataset)),sigma=row['sigma'],clip=row['clip'])
            dl = DataLoader(ds,pin_memory=True,shuffle=False)
            res=trainer.test(model=dn,test_dataloaders=[dl],verbose=False)
            print(res)
            top_networks_now.loc[top_networks_now['ID']==row['ID'],test_dataset] = res[0]['test_psnr']
            
            
    top_networks_now.to_csv('bm3d_with_test_results.csv')


    


