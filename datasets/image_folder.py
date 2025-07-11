import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from datasets import register

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

@register('image-folder')
class ImageFolder(Dataset):
    def __init__(self, path,  split_file=None, split_key=None, first_k=None, size=None,
                 repeat=1, cache='none', mask=False,ignore_bg = False):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key

        self.size = size
        self.mask = mask

        self.filenames = []  # 新增：存储文件名列表

        if self.mask:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),#, interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []

        for filename in filenames:
            file = os.path.join(path, filename)
            self.append_file(file)
            self.filenames.append(filename)  # 新增：保存文件名

    def append_file(self, file):
        if self.cache == 'none':
            self.files.append(file)
        elif self.cache == 'in_memory':
            self.files.append(self.img_process(file))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        filename = self.filenames[idx % len(self.files)]  # 新增：获取对应的文件名
        
        if self.cache == 'none':
            img = self.img_process(x)
            return img, filename  # 只返回图像和文件名
        elif self.cache == 'in_memory':
            return x, filename  # 只返回图像和文件名

    def img_process(self, file):
        return Image.open(file).convert('RGB')  # 只返回图像对象  
    



@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, classes, palette,**kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs, mask=True)
        self.n_classes = len(classes)
        #self.classes =['building','vegetation','water','road']     
        #self.palette = [ [1,0,0], [0,1,0], [0,0,1], [1,1,0]] #background,building,vegetation,water,road
        
        #self.classes =['building','vegetation','water','road','background']
        #self.palette = [ [1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,0,0]] 

        self.classes = classes
        self.palette = palette


    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        img, filename1 = self.dataset_1[idx]
        mask, filename2 = self.dataset_2[idx]
        assert filename1 == filename2, f"Filenames do not match: {filename1} vs {filename2}"
        return img, mask, filename1 # 修改：返回图像、掩码和文件名
    