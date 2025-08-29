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
                 repeat=1, cache='none', mask=False,ignore_bg = False, file_extension='.jpg'):
        self.repeat = repeat
        self.cache = cache
        self.path = path
        self.Train = False
        self.split_key = split_key
        self.file_extension = file_extension  # 新增：文件扩展名

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
                transforms.Resize((self.size, self.size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.430, 0.411, 0.296],
                                     std=[0.213, 0.156, 0.143])
            ])

        if split_file is None:
            filenames = sorted(os.listdir(path))
        else:
            with open(split_file, 'r') as f:
                content = f.read().strip()
                # 尝试解析为JSON
                try:
                    data = json.loads(content)
                    if split_key and split_key in data:
                        filenames = data[split_key]
                    else:
                        # 如果没有指定split_key或key不存在，使用第一个值
                        filenames = list(data.values())[0] if data else []
                except json.JSONDecodeError:
                    # 如果不是JSON格式，按行分割
                    filenames = [line.strip() for line in content.split('\n') if line.strip()]
                
                # 如果文件名没有扩展名，添加默认扩展名
                processed_filenames = []
                for filename in filenames:
                    if not os.path.splitext(filename)[1]:  # 没有扩展名
                        filename = filename + self.file_extension
                    processed_filenames.append(filename)
                filenames = processed_filenames
                
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


@register('paired-image-folders-multitask')
class PairedImageFoldersMultiTask(Dataset):
    """支持多任务的配对图像文件夹数据集"""
    
    def __init__(self, root_path_1, root_path_2, classes, palette, task_id=0, 
                 split_file=None, split_key=None, file_extension='.jpg', **kwargs):
        # 创建支持split_file的ImageFolder实例
        self.dataset_1 = ImageFolder(root_path_1, split_file=split_file, 
                                    split_key=split_key, file_extension=file_extension, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, split_file=split_file, 
                                    split_key=split_key, mask=True, file_extension=file_extension, **kwargs)
        self.n_classes = len(classes)
        self.classes = classes
        self.palette = palette
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        img, filename1 = self.dataset_1[idx]
        mask, filename2 = self.dataset_2[idx]
        assert filename1 == filename2, f"Filenames do not match: {filename1} vs {filename2}"
        return img, mask, filename1, self.task_id
    