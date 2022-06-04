"""
datasets.py

Datasets for AWGN denoising

"""

import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import make_grid
from matplotlib.pyplot import imread,imshow,subplots,show,pause
from pathlib import Path
from torchvision import transforms
from torchvision.transforms import RandomCrop,ToTensor,RandomResizedCrop,RandomVerticalFlip,RandomHorizontalFlip
from pathlib import Path
from PIL import Image
import cv2
import glob
import h5py

# show grid of images
def show_grid(img,ax):
    img = img.clamp(min=0,max=1)
    img = make_grid(img)
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

# add noise with the given sigma value
def get_noisy_img(img,sigma,clip=True): 
    noisy = img+(torch.randn(img.shape)*(sigma))
    if clip: noisy.clamp_(min=0,max=1)
    return noisy

# show random images from dataset
def show_images(dl:DataLoader,num_batches:int,model=None)->None:
    for batch_idx,(noisy,clean) in enumerate(dl):
        
        if model is None: fig,ax = subplots(2,1) # Only show noisy,clean pairs
        else: fig,ax = subplots(3,1) # show noisy,estimated,clean

        show_grid(noisy,ax[0])
        if model is None: show_grid(clean,ax[1])
        else: 
            show_grid(model(noisy),ax[1])
            show_grid(clean,ax[2])
        
        show()
        
        if batch_idx>=num_batches: break

class SIDDDataset(Dataset):
    def __init__(self,patches_per_batch=32,patches_per_image=512,patch_size=80,path=None):
        print('Initializing SIDD dataset')
        self.path = 'datasets/train/sidd_patches/sidd_medium_80_{0}/part{1}.h5'
        self.patches_per_batch = patches_per_batch
        self.patches_per_image = patches_per_image
        self.image_to_batch_ratio = self.patches_per_image//self.patches_per_batch

    def __len__(self):
        return (320*self.patches_per_image)//self.patches_per_batch

    def __getitem__(self,index):
        hf = h5py.File(self.path.format(self.patches_per_image,index//self.image_to_batch_ratio), 'r')
        #except: print(index,self.patches_per_image,self.patches_per_batch)
        data = np.asarray(hf['patches'])
        hf.close()

        #indices = np.random.randint(low=0,high=self.patches_per_image,size=(self.patches_per_batch,))
        start,stop = (index%self.image_to_batch_ratio)*self.patches_per_batch,(index%self.image_to_batch_ratio+1)*self.patches_per_batch
        data = torch.tensor(data[start:stop,:,:,:])
        noisy,clean = data[:,0],data[:,1]

        noisy = noisy.float().div(255).permute(0,3,1,2)
        clean = clean.float().div(255).permute(0,3,1,2)

        return (
            noisy,
            clean
        )

class AWGNDataset(Dataset):
    def __init__(self,path="datasets/train/Pascal1500/Color/",num_images=1000,sigma=25,clip=True,num_channels=3):
        super().__init__()
        self.path = path+'pascal_{0}.jpg'
        self.len = num_images
        self.sigma = sigma/255.
        self.clip = clip
        self.num_channels = num_channels
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        img = imread(self.path.format(index))
        img = torch.tensor(np.copy(img)).float()
        if img.max()>1: img.div_(255.)
        img = img.permute(2,0,1)[:self.num_channels,:,:] # channels first
        return get_noisy_img(img,self.sigma,self.clip),img # generate noisy image and return

class TestDataset(Dataset):
    def __init__(self,path="datasets/test/Kodak/",sigma=25,clip=True,num_channels=3):
        super().__init__()
        self.path = str(Path(path).joinpath('*'))
        self.images = glob.glob(self.path)
        self.len = len(self.images)
        self.sigma = sigma/255.
        self.clip = clip
        self.num_channels = num_channels
        torch.manual_seed(0)
    def __len__(self):
        return self.len
    def __getitem__(self,index):
        clean = imread(self.images[index])
        clean = torch.tensor(np.copy(clean)).float().permute(2,0,1)[:self.num_channels,:,:]
        if clean.max()>1: clean.div_(255.)
        noisy = clean+(torch.randn(clean.shape).mul_(self.sigma))
        if self.clip: noisy.clamp_(min=0,max=1)
        return noisy,clean


class BigDataset(Dataset):
    def __init__(self,paths=[],sigma=25,clip=True,num_channels=3):
        super().__init__()
        self.images = []
        if paths==[]:
            paths =[
                '/scratch/zhangh/malik/DENOISING/WATERLOO',
                '/scratch/zhangh/malik/DENOISING/DIV2K',
                '/scratch/zhangh/malik/DENOISING/Pascal_VOC_2007'
            ]
        for path in paths: self.images += glob(str(Path(path).joinpath('*')))
        self.len = len(self.images)
        self.sigma = sigma/255.
        self.clip = clip
        self.composed = transforms.Compose([RandomCrop(60)])
        self.num_channels = num_channels
    
    def transform(self,x):
        return self.composed(x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        clean = imread(self.images[index])
        clean = torch.tensor(np.copy(clean)).float().permute(2,0,1)[:self.num_channels,:,:]
        clean = self.transform(clean)
        if clean.max()>1: clean.div_(255.)
        noisy = clean+(torch.randn(clean.shape).mul_(self.sigma))
        if self.clip: noisy.clamp_(min=0,max=1)
        return noisy,clean

class BigDatasetWithAugmentation(Dataset):
    def __init__(self,paths=[],sigma=25,clip=True,num_channels=3):
        super().__init__()
        self.images = []
        if paths==[]:
            paths =[
                '/scratch/zhangh/malik/DATASETS/WATERLOO',
                '/scratch/zhangh/malik/DATASETS/DIV2K',
                '/scratch/zhangh/malik/DATASETS/Pascal_VOC_2007'
            ]
        for path in paths: self.images += glob(str(Path(path).joinpath('*')))
        self.len = len(self.images)
        self.sigma = sigma/255.
        self.clip = clip
        self.num_channels = num_channels
        self.composed = transforms.Compose([RandomResizedCrop(60,interpolation=Image.BICUBIC),RandomVerticalFlip(),RandomHorizontalFlip()])
    
    def transform(self,x):
        return self.composed(x)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,index):
        clean = imread(self.images[index])
        clean = torch.tensor(np.copy(clean)).float().permute(2,0,1)[:self.num_channels,:,:]
        clean = self.transform(clean)
        if clean.max()>1: clean.div_(255.)
        noisy = clean+(torch.randn(clean.shape).mul_(self.sigma))
        if self.clip: noisy.clamp_(min=0,max=1)
        return noisy,clean

class DenoisingDatasetDnCNN(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, sigma, clip=False,num_channels=1):
        super(DenoisingDatasetDnCNN, self).__init__()
        self.xs = datagenerator()
        self.xs = self.xs.astype('float32')/255.0
        self.xs = torch.from_numpy(self.xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
        self.sigma = sigma
        self.clip = clip
        self.num_channels = num_channels

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        if self.clip: batch_y.clamp_(min=0,max=1)
        batch_x = batch_x[:self.num_channels,:,:]
        batch_y = batch_y[:self.num_channels,:,:]
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)



def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name):
    patch_size, stride = 40, 10
    aug_times = 1
    scales = [1, 0.9, 0.8, 0.7]
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='/scratch/zhangh/malik/BM3D_VS_SELFONN/datasets/train/BSD400', verbose=False):
    batch_size = 128
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data

if __name__ == "__main__":
    from models import BM3D
    ds = BigDatasetWithAugmentation(sigma=30,clip=False,num_channels=3)
    dl = DataLoader(ds,batch_size=1)
    show_images(dl,8)