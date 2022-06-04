import torch
from torch.utils.data import Dataset,DataLoader,Subset,random_split
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from .models import SelfONN,CNN,BM3D,get_model
from .metrics import BatchPSNR
from .datasets import AWGNDataset,show_grid,subplots,show,pause,TestDataset,BigDataset,BigDatasetWithAugmentation,DenoisingDatasetDnCNN,SIDDDataset


class Denoiser(pl.LightningModule):
    def __init__(self,model,num_layers,num_channels=1,num_neurons=25,sigma=5,clip=False,train_ratio=0.9,q=1,max_shifts=[0,10,10],rounded_shifts=False,dataset_name='big_augment'):
        super().__init__()
        self.model = model(num_channels=num_channels,num_neurons=num_neurons,num_layers=num_layers,q=q,max_shifts=max_shifts,rounded_shifts=rounded_shifts)
        self.val_psnr = BatchPSNR()
        self.test_psnr = BatchPSNR()
        self.sigma = sigma
        self.clip = clip
        self.train_ratio = train_ratio
        self.dataset_name = dataset_name
        self.num_channels = num_channels
        self.max_shifts = max_shifts
        self.rounded_shifts = rounded_shifts
        self.save_hyperparameters('num_layers','num_channels','num_neurons','sigma','clip','train_ratio','q','dataset_name','max_shifts','rounded_shifts')

    def setup(self,stage):
        if stage=='train':
            if self.dataset_name == 'small': ds = AWGNDataset(sigma=self.sigma,clip=self.clip,num_channels=self.num_channels)
            elif self.dataset_name=='big': ds = BigDataset(sigma=self.sigma,clip=self.clip,num_channels=self.num_channels)
            elif self.dataset_name=='big_augment': ds = BigDatasetWithAugmentation(sigma=self.sigma,clip=self.clip,num_channels=self.num_channels)
            elif self.dataset_name=='dncnn': ds = DenoisingDatasetDnCNN(sigma=self.sigma,clip=self.clip,num_channels=self.num_channels)
            elif self.dataset_name=='sidd': ds = SIDDDataset()
            else: raise NotImplementedError

            train_len = round(len(ds)*self.train_ratio)
            val_len = len(ds)-train_len
            self.train_ds,self.val_ds = random_split(ds,[train_len,val_len],generator=torch.Generator().manual_seed(42))
            
            print("Train:",train_len)
            print("Val:",val_len)
            #print("Test:",len(self.test_ds))
        
        self.top_val_psnr = -1e9

    def training_step(self, batch, batch_idx):
        x, y = batch
        if x.dim()==5:
            _, _, c, h, w = x.size()
            x = x.view(-1,c,h,w)
            y = y.view(-1,c,h,w)
        y_est = self.model(x)
        loss = F.mse_loss(y_est, y,reduction='sum')
        return loss
    
    def training_epoch_end(self,losses):
        self.log('train_loss',torch.stack([x['loss'] for x in losses]).mean())
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if x.dim()==5:
            _, _, c, h, w = x.size()
            x = x.view(-1,c,h,w)
            y = y.view(-1,c,h,w)
        y_est = self.model(x)
        self.val_psnr.update(y_est,y)
        return y_est
        
    def validation_epoch_end(self,outs):
        val_psnr = self.val_psnr.compute()
        self.top_val_psnr = max(self.top_val_psnr,val_psnr)
        self.log('top_val_psnr',self.top_val_psnr)
        self.val_psnr.reset()

    def on_test_epoch_start(self):
        self.test_psnr = BatchPSNR().to(self.device)
        self.test_psnr.reset()
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_est = self.model(x)
        self.test_psnr.update(y_est,y)
        return y_est
        
    def test_epoch_end(self,outs):
        self.log('test_psnr',self.test_psnr.compute())
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def train_dataloader(self): 
        num_workers = 40
        return DataLoader(self.train_ds, batch_size=8,pin_memory=True,num_workers=num_workers,shuffle=True)
    
    def val_dataloader(self): 
        num_workers = 40
        return DataLoader(self.val_ds, batch_size=8,pin_memory=True,num_workers=num_workers)
    

class Denoiser_BM3D(pl.LightningModule):
    def __init__(self,num_channels=1,sigma=5,clip=False):
        super().__init__()
        self.model = BM3D(num_channels=num_channels,sigma=sigma)
        
    def on_test_epoch_start(self):
        self.test_psnr = BatchPSNR().to(self.device)
        self.test_psnr.reset()
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_est = self.model(x)
        self.test_psnr.update(y_est,y)
        return y_est
        
    def test_epoch_end(self,outs):
        test_psnr = self.test_psnr.compute()
        self.test_psnr.reset()
        return {'test_psnr':test_psnr}
        
    