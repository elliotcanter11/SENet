import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import torch
from skimage.color import rgb2lab
random.seed(3407)


class RGBtoRGBLAB(object):
    """Convert RGB tensor to concatenated RGB+LAB tensor"""
    def __call__(self, img):
        # img shape: [3, H, W], values in [0, 1]
        rgb = img.numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
        
        # Convert to LAB
        lab = rgb2lab(rgb)  # L:[0,100], a:[-127,127], b:[-127,127]
        
        # Normalize LAB to [0, 1]
        lab_norm = np.stack([
            lab[..., 0] / 100.0,
            (lab[..., 1] + 128) / 256.0,
            (lab[..., 2] + 128) / 256.0
        ], axis=-1)
        
        # Concatenate [RGB, LAB] and convert to tensor
        six_ch = np.concatenate([rgb, lab_norm], axis=2)
        return torch.from_numpy(six_ch.transpose(2, 0, 1)).float()


rgb_mean = torch.tensor([0.434, 0.424, 0.327]).view(3,1,1)
rgb_std  = torch.tensor([0.254, 0.242, 0.247]).view(3,1,1)

lab_mean = torch.tensor([0.447, 0.493, 0.553]).view(3,1,1)
lab_std  = torch.tensor([0.246, 0.052, 0.077]).view(3,1,1)

class NormalizeRGBLAB(object):
    """Normalize RGB and LAB parts separately"""
    def __call__(self, img):
        # img shape: [6, H, W], values in [0, 1]
        rgb = img[:3]
        lab = img[3:]
        
        # Custom normalization for RGB
        rgb = (rgb - rgb_mean) / rgb_std
        
        # Custom normalization for LAB
        lab = (lab - lab_mean) / lab_std
        
        return torch.cat([rgb, lab], dim=0)


class make_Dataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
    
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')] #img path list
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]          #mask path list
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.size = len(self.images)      #length of dataset

        self.img_transform = transforms.Compose([                            #对图片进行预处理
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            RGBtoRGBLAB(),
            NormalizeRGBLAB()])
 
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            # transforms.Resize((trainsize, trainsize), interpolation=Image.NEAREST), #用这种插值方式gt里面只有0和1
            transforms.ToTensor()])
        
    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        self.getFlip()
        image = self.flip1(image)
        gt = self.flip1(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # weit = gt
        return image, gt#, weit

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')#转为灰度图
        
    def getFlip(self):
        p1 = random.randint(0, 1)
        self.flip1 = transforms.RandomHorizontalFlip(p1)

    def __len__(self):
        return self.size

class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, gt_root, testsize):

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            RGBtoRGBLAB(),
            NormalizeRGBLAB()])
        
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def get_loader(image_root, gt_root, 
               batchsize = 16, trainsize=384, shuffle=True, num_workers=12, pin_memory=True):
    # `num_workers=0` for more stable training
    dataset = make_Dataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                #   drop_last = True,
                                  pin_memory=pin_memory)

    return data_loader
