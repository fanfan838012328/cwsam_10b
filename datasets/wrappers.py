import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn.functional as F

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    mask = mask.permute(1,2,0)*255
    mask.int()
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = torch.all(equality, dim=-1)
        semantic_map.append(class_map)
    
    # bg_equality = np.equal(mask, [0,0,0])
    # bg_map = torch.all(bg_equality, dim=-1)
    # semantic_map[1] += bg_map

    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = torch.as_tensor(semantic_map)
    semantic_map = semantic_map.permute(2,0,1)
    map=torch.sum(semantic_map,dim=0)
    #assert map.all()
    return semantic_map

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1,2,0)
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x=x.permute(2,0,1)
    return x

def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))
    # return transforms.ToTensor()(
    #     #transforms.Resize(size)(
    #     transforms.ToPILImage()(mask))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False,**kwargs):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment


        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # img, mask, filename = self.dataset[idx]
        img, mask,_= self.dataset[idx]
        
        # 确保 img 和 mask 都是 PIL.Image 对象
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image for img, got {type(img)}")
        if not isinstance(mask, Image.Image):
            raise TypeError(f"Expected PIL.Image for mask, got {type(mask)}")
        
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = mask_to_onehot(mask, self.dataset.palette)

    
        return {
            'inp': img,
            'gt': mask,
            # 'file_name': filename
        }
@register('test')
class TestDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False,**kwargs):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment


        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask, filename = self.dataset[idx]
        # img, mask,_= self.dataset[idx]
        
        # 确保 img 和 mask 都是 PIL.Image 对象
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image for img, got {type(img)}")
        if not isinstance(mask, Image.Image):
            raise TypeError(f"Expected PIL.Image for mask, got {type(mask)}")
        
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = mask_to_onehot(mask, self.dataset.palette)

    
        return {
            'inp': img,
            'gt': mask,
            'file_name': filename
        }
@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask,_ = self.dataset[idx]

        # random filp
        # if random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)
        mask = self.mask_transform(mask)
        mask = mask_to_onehot(mask,self.dataset.palette)

        return {
            'inp': self.img_transform(img),
            #'gt': self.mask_transform(mask)
            'gt': mask
        }


def mask_to_onehot_aug(mask, palette):
    """
    Converts a segmentation mask (H, W) to (K, H, W) where K is the number of classes.
    """
    mask = np.array(mask)
    num_classes = len(palette)
    one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)

    for i, color in enumerate(palette):
        one_hot[i] = np.all(mask == color, axis=-1)

    return torch.from_numpy(one_hot)

def mask_to_onehot_optimized(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (K, H, W) where K is the number of classes,
    using NumPy broadcasting and vectorization for faster performance.
    """
    mask = np.array(mask, dtype=np.int32)  # 确保 mask 是整数类型
    palette = np.array(palette, dtype=np.int32)  # 同上
    mask = mask[:, :, None]  # 将 mask 从 (H, W, C) 扩展到 (H, W, 1, C)
    palette = palette[None, None, :, :]  # 将 palette 从 (K, C) 扩展到 (1, 1, K, C)

    # 使用广播比较所有颜色和掩码，并累计沿颜色通道的结果
    one_hot = np.all(
        mask == palette, axis=3
    )  # 进行比较并累计结果，得到形状为 (H, W, K)

    return torch.from_numpy(one_hot.transpose(2, 0, 1))  # 重新排序维度以匹配 (K, H, W)

@register('train_aug')
class TrainAugDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=True, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.inp_size = inp_size

        # 定义Albumentations变换
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=180, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.RandomResizedCrop(height=1024, width=1024, scale=(0.5, 1.0), p=0.5),
                A.CoarseDropout(max_holes=10, max_height=10, max_width=10, p=0.5),
                A.RandomGridShuffle(grid=(4, 4), p=0.5),
                A.Blur(blur_limit=5, p=0.5),
                A.Resize(height=1024, width=1024),  # 确保最终大小为1024
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                 std=[1, 1, 1])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img = np.array(img)
        mask = np.array(mask)

        # 应用数据增强
        if self.augment:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # mask.permute(2,0,1)
        mask = mask_to_onehot_optimized(mask, self.dataset.palette).to(img.dtype)

        # 可视化 img 和 mask
        
        return {
            'inp': img,
            'gt': mask
        }

    def visualize(self, idx):
        import matplotlib.pyplot as plt
        data = self.__getitem__(idx)
        img = data['inp'].permute(1, 2, 0).numpy()  # C, H, W -> H, W, C
        mask = data['gt'].permute(1, 2, 0).numpy()  # C, H, W -> H, W, C
        mask = np.argmax(mask, axis=2)  # 取最大索引

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')

        plt.show()
