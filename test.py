import torch
from utils.denoiser import Denoiser
from utils.models import get_model
from utils.datasets import TestDataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm

def get_top_networks(csv_path = "wandb_export_2021-01-09T19_44_21.092+02_00.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1)
    splitted = df['Name'].str.split('_',n=1,expand=True)
    df['Run Name'] = splitted[0]
    df['Version'] = splitted[1]
    df = df.drop(columns=['Name'])
    idx = df.groupby(['Run Name'])['top_val_psnr'].transform(max) == df['top_val_psnr']
    df = df[idx]
    print(df)
    top_networks = df[['ID','Run Name','Version','num_neurons','q','sigma','top_val_psnr','clip']]
    return top_networks
    

if __name__ == '__main__':

    torch.manual_seed(0)

    # test_path
    test_path = Path("datasets/test")
    test_datasets = ['Kodak','McM','CBSD68']
        
    # load trained models and evaluate dataloader
    project_name = 'bm3d_vs_selfonn'
    ckpt_path = "wandb\\*{id}\\files\\{project}\\{id}\\checkpoints\\*.ckpt"
    top_networks = get_top_networks('wandb_latest.csv')
    for index, row in (top_networks.iterrows()):
        # load model from checkpoint
        model = get_model('selfonn') # big messup this..
        dn = Denoiser.load_from_checkpoint(glob.glob(ckpt_path.format(id=row['ID'],project=project_name))[0],model=model)
        trainer = pl.Trainer(gpus=1,weights_summary=None,progress_bar_refresh_rate=0)

        # load datasets and evaluate
        for test_dataset in test_datasets:
            print(test_dataset)    
            ds = TestDataset(str(test_path.joinpath(test_dataset)),sigma=row['sigma'],clip=row['clip'])
            dl = DataLoader(ds,pin_memory=True,shuffle=False)
            res=trainer.test(model=dn,test_dataloaders=[dl],verbose=False)
            print(res)
            top_networks.loc[top_networks['ID']==row['ID'],test_dataset] = res[0]['test_psnr']
            top_networks.to_csv('top_networks_with_test_results.csv')


    


