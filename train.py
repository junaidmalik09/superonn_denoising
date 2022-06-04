import torch
from utils.denoiser import Denoiser
from utils.models import get_model
from utils.datasets import TestDataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from utils.misc import get_random_alphanumeric_string
from pathlib import Path


def train(model_name,clip,train_ratio,q,sigma,num_layers,num_neurons,num_channels,max_epochs,gpus,dataset_name,max_shifts,rounded_shifts,runs=3,project_name='bm3d_vs_selfonn',wandb=True):

    name=get_random_alphanumeric_string()
    
    for version in range(runs):
        
        
        name_now = name+'_'+str(version)

        checkpoint_callback = ModelCheckpoint(
            monitor='top_val_psnr',
            mode='max',
            verbose=False,
        )

        # Init
        wandb_logger = WandbLogger(
            name=name_now,
            project=project_name,
            log_model=False
        ) if wandb else None
        
        # init model
        model = get_model(model_name)
        dn = Denoiser(
            model=model,
            num_layers=num_layers,
            num_channels=num_channels,
            num_neurons=num_neurons,
            sigma=sigma,
            clip=clip,
            train_ratio=train_ratio,
            q=q,
            dataset_name=dataset_name,
            max_shifts=max_shifts,
            rounded_shifts=rounded_shifts
        )

        # Initialize a trainer
        trainer = pl.Trainer(
            gpus=gpus, 
            max_epochs=max_epochs,
            logger=wandb_logger,
            checkpoint_callback=checkpoint_callback
        )

        # Train the model
        trainer.fit(dn)


        # Test
        test_path = Path("datasets/test")
        test_datasets = ['Kodak','McM','CBSD68']
        test_results = {}
        for test_dataset in test_datasets:
            print(test_dataset)    
            ds = TestDataset(str(test_path.joinpath(test_dataset)),sigma=sigma,clip=clip,num_channels=num_channels)
            dl = DataLoader(ds,pin_memory=True,shuffle=False)
            r = trainer.test(test_dataloaders=[dl],verbose=False)
            test_results[test_dataset] = r[0]['test_psnr']
        
        wandb_logger.log_metrics(test_results)

        if wandb: 
            wandb_logger.finalize('Done')
            wandb_logger.experiment.finish()



if __name__ == "__main__":

    # argument parser
    parser = ArgumentParser()
    parser.add_argument('--max_epochs',type=int,default=100)
    parser.add_argument('--gpus',type=int, default=1)
    parser.add_argument('--num_channels',type=int,default=3)
    parser.add_argument('--num_neurons',nargs="+",type=int,default=[1024])
    parser.add_argument('--num_layers',nargs="+",type=int,default=[17])
    parser.add_argument('--dataset_name',type=str,default='sidd')
    parser.add_argument('--sigma',nargs="+",type=int,default=[0])
    parser.add_argument('--q',nargs="+",type=int,default=[1])
    parser.add_argument('--train_ratio',nargs="+",type=int,default=[0.95])
    parser.add_argument('--model_name',type=str,default='dncnn')
    parser.add_argument('--clip',nargs="+",type=int,default=[1])
    parser.add_argument('--wandb',type=bool,default=True)
    parser.add_argument('--max_shifts',nargs="+",type=int,default=[0,10,10])
    parser.add_argument('--rounded_shifts',type=int,default=0)


    args = parser.parse_args()

    for clip in args.clip:
        for train_ratio in args.train_ratio:
            for num_layers in args.num_layers:
                for num_neurons in args.num_neurons:
                    for sigma in args.sigma:
                        for q in args.q:
                            print(args.dataset_name,args.model_name,clip==1,train_ratio,sigma,num_layers,num_neurons,q)
                            train(
                                args.model_name,
                                clip==1,
                                train_ratio,
                                q,
                                sigma,
                                num_layers,
                                num_neurons,
                                args.num_channels,
                                args.max_epochs,
                                args.gpus,
                                args.dataset_name,
                                args.max_shifts,
                                args.rounded_shifts==1,
                                runs=1,
                                project_name='bm3d_vs_superonn',
                                wandb=args.wandb
                            )
                            print("="*64)
