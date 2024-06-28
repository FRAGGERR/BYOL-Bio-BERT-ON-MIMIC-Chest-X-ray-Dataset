from dataclasses import dataclass
from functools import partial
import random
from typing import Tuple, Union

from PIL import ImageFilter, Image
import torch
from torch import Tensor
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import ImageFilter
import copy
import numpy as np
from torchvision.transforms import InterpolationMode

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
    
class PCRLv2Augmentation:
    def __init__(self,config, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        n_class    = config['data']['n_class']
        image_size = config['data']['resize_size']
        
        self.num_local_view = config['data']['num_local_view']
        
        self.spatial_transform = T.Compose([
                                            T.RandomResizedCrop(224, scale=(0.3, 1)),
                                            T.RandomRotation(10),
                                            T.RandomHorizontalFlip()
                                        ])
        self.spatial_transform_local = T.Compose([
                                            T.RandomResizedCrop(96, scale=(0.05, 0.3)),
                                            T.RandomRotation(10),
                                            T.RandomHorizontalFlip()
                                        ])
        self.train_transform = T.Compose([
                                            T.RandomGrayscale(p=0.2),
                                            T.RandomApply([GaussianBlur()], p=0.5),
                                            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                            T.ToTensor(),
                                            T.Normalize(mean=mean, std=std)
                                        ])
        self.local_transform = T.Compose([
                                            T.RandomGrayscale(p=0.2),
                                            T.RandomApply([GaussianBlur()], p=0.5),
                                            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                            T.ToTensor(),
                                            T.Normalize(mean=mean, std=std)
                                        ])
        self.train_transform.transforms.append(Cutout(n_holes=3, length=32))
        
        self.normalize_trans = T.Compose([T.ToTensor(),
                                          T.Normalize(mean=mean, std=std)])
        
        
    def __call__(self, img: Tensor):
        y1 = self.spatial_transform(img)
        y2 = self.spatial_transform(img)
        
        norm_y1 = self.normalize_trans(y1)
        norm_y2 = self.normalize_trans(y2)
        
        x1 = copy.deepcopy(norm_y1)
        x2 = copy.deepcopy(norm_y2)
        
        y1 = self.train_transform(y1)
        y2 = self.train_transform(y2)
        
        local_views = []
        for i in range(self.num_local_view):
            local_view = self.spatial_transform_local(img)
            local_view = self.local_transform(local_view)
            local_views.append(local_view)
            
        return [y1, y2, x1, x2, local_views]
    
    
    
class Custom_Distoration(object):
    def __init__(self, input_rows=224, input_cols=224):
        self.input_rows = input_rows
        self.input_cols = input_cols
    def __call__(self, org_img):
        org_img = np.array(org_img)
        r = random.random()
        if r <= 0.3:  #cut-out
            cnt = 10
            while cnt > 0:
                block_noise_size_x, block_noise_size_y = random.randint(10, 70), random.randint(10, 70)
                noise_x, noise_y = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                org_img[noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,:] = 0
                cnt = cnt - 1
                if random.random() < 0.1:
                    break
        elif 0.3 < r <= 0.35:  #cut-out
            image_temp = copy.deepcopy(org_img)
            org_img[:, :,:] = 0
            cnt = 10
            while cnt > 0:
                block_noise_size_x, block_noise_size_y = self.input_rows - random.randint(50,70), self.input_cols - random.randint(50, 70)
                noise_x, noise_y = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                org_img[noise_x:noise_x + block_noise_size_x,noise_y:noise_y + block_noise_size_y,:] = image_temp[noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y,:]
                cnt = cnt - 1
                if random.random() < 0.1:
                    break
        elif 0.35 < r <= 0.65:  #shuffling
            cnt = 10
            image_temp = copy.deepcopy(org_img)
            while cnt > 0:
                while True:
                    block_noise_size_x, block_noise_size_y = random.randint(10, 15), random.randint(10, 15)
                    noise_x1, noise_y1 = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                    noise_x2, noise_y2 = random.randint(3, self.input_rows - block_noise_size_x - 3), random.randint(3,self.input_cols - block_noise_size_y - 3)
                    if ((noise_x1 > noise_x2 + block_noise_size_x) or (noise_x2 > noise_x1 + block_noise_size_x) or (noise_y1 < noise_y2 + block_noise_size_y) or (noise_y2 < noise_y1 + block_noise_size_y)):
                         break

                org_img[noise_x1:noise_x1 + block_noise_size_x, noise_y1:noise_y1 + block_noise_size_y,:] = image_temp[noise_x2:noise_x2 + block_noise_size_x,noise_y2:noise_y2 + block_noise_size_y,:]
                org_img[noise_x2:noise_x2 + block_noise_size_x, noise_y2:noise_y2 + block_noise_size_y,:] = image_temp[noise_x1:noise_x1 + block_noise_size_x,noise_y1:noise_y1 + block_noise_size_y,:]
                cnt = cnt - 1
        return Image.fromarray(org_img)

class CAiDAugmentation:
    def __init__(self,config):
        self.crop_transform=T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.))])
        self.transform = T.Compose([T.RandomApply([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        T.RandomHorizontalFlip()])
        self.reconstruction_transform=T.Compose([
            Custom_Distoration(224,224),
            T.ToTensor(),
        ])
        self.to_tensor = T.Compose([
            T.ToTensor()
        ])

    def __call__(self, x):
        y1_orig = self.crop_transform(x)
        y2_orig = self.crop_transform(x)
        y1 = self.transform(y1_orig)
        y2 = self.transform(y2_orig)
        
        y1 = self.reconstruction_transform(y1)
        y2 = self.reconstruction_transform(y2)
        y1_orig_1c = y1_orig.convert('L')
        y1_orig_1c = np.array(y1_orig_1c) / 255.0
        y1_orig_1c = np.expand_dims(y1_orig_1c, axis=0).astype('float32')
        return [y1, y2, y1_orig_1c]
    

class BYOLAugmentations:
    def __init__(self,config): #
        image_size   = config['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size ,scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0) ,interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.8),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=1.0),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
                ])
        self.augment_prime = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.1, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.5),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomSolarize(threshold=128, p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
                ])
        
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment_prime(x)
        return [x1,x2]   
    
    
class BarlowAugmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomSolarize(threshold=128, p=0.0),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
        self.augment_prime = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomSolarize(threshold=128, p=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment_prime(x)
        return [x1,x2]

    
    
class VICRegAUgmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=1.0),
                T.RandomSolarize(threshold=128, p=0.0),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
        self.augment_prime = T.Compose([
                T.RandomResizedCrop(image_size, interpolation=InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.1),
                T.RandomSolarize(threshold=128, p=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
        
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment_prime(x)
        return [x1,x2]


class SimSiamAUgmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
                ])
        
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment(x)
        return [x1,x2]
    
class SimCLRAUgmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
                ])
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment(x)
        return [x1,x2]

    
    
class MoCoAUgmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomGrayscale(p=0.2),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),                
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
                ])
        
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment(x)
        return [x1,x2]
    
    
class MoCov2AUgmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.augment = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
                ])
        
    def __call__(self, x):
        x1 = self.augment(x)
        x2 = self.augment(x)
        return [x1,x2]

    
        
class MoCov3AUgmentations:
    def __init__(self,config):
        image_size   = config['data']['resize_size']
        mean  = [0.485, 0.456, 0.406]
        std   = [0.229, 0.224, 0.225]
        self.crop_min = config['data']['crop_min']
        
        self.augment1 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(self.crop_min,1.0)),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1,sigma=(0.1, 2.0))], p=1.0),
                T.RandomHorizontalFlip(),
                T.ToTensor(),          
                T.Normalize(mean=mean, std=std)
                ])
        
        self.augment2 = T.Compose([
                T.RandomResizedCrop(image_size, scale=(self.crop_min,1.0)),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1,sigma=(0.1, 2.0))], p=0.1),
                T.RandomSolarize(threshold=128, p=0.2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),          
                T.Normalize(mean=mean, std=std)
                ])
        
    def __call__(self, x):
        x1 = self.augment1(x)
        x2 = self.augment2(x)
        return [x1,x2]


                
        
class SiMEXAugmentation:
    def __init__(self,config, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        image_size          = config['data']['resize_size']        
        self.num_local_view = config['data']['num_local_view']
        n_cb           = config['model']['number_of_com_branches']
        self.total_global_views = n_cb+1
        self.total_local_views  = n_cb
        
        self.global_transform = T.Compose([
                                            T.RandomResizedCrop(image_size, scale=(0.3, 1)),
                                            T.RandomRotation(10),
                                            T.RandomHorizontalFlip(),
                                            T.RandomGrayscale(p=0.2),
                                            T.RandomApply([GaussianBlur()], p=0.5),
                                            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                            T.ToTensor(),
                                            T.Normalize(mean=mean, std=std)
                                        ])
        
        self.local_transform = T.Compose([
                                            T.RandomResizedCrop(int(image_size*3/7), scale=(0.05, 0.14)),
                                            T.RandomRotation(10),
                                            T.RandomHorizontalFlip(),
                                            T.RandomGrayscale(p=0.2),
                                            T.RandomApply([GaussianBlur()], p=0.5),
                                            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                            T.ToTensor(),
                                            T.Normalize(mean=mean, std=std)
                                        ])
        
               
    def __call__(self, img):
        Global_views = []
        Local_views = []
        
        # Generate global views for each branch
        for _ in range(self.total_global_views):
            global_view = self.global_transform(img)
            Global_views.append(global_view)
        
        # Generate local views for each branch
        for _ in range(self.total_local_views):
            local_views = []
            for _ in range(self.num_local_view):
                local_view = self.local_transform(img)
                local_views.append(local_view)
            Local_views.append(local_views)
            
        return [Global_views, Local_views] 
        

        
        
def divide_lung_into_parts(right_lung_coords, left_lung_coords):

    def divide_single_lung(lung_coords):
        x, y, width, height = lung_coords        
        midline_x = x + width // 2
        
        upper_y  = y
        middle_y = upper_y + height // 3
        lower_y  = middle_y + height // 3
        
        return {'upper':  [x, upper_y, width, height // 3],
                'middle': [x, middle_y, width, height // 3],
                'lower':  [x, lower_y, width, height // 3]}
    
    
    # Divide the right lung into parts if coordinates are provided
    right_lung_parts = None
    if right_lung_coords is not None:
        right_lung_parts = divide_single_lung(right_lung_coords)
    
    # Divide the left lung into parts if coordinates are provided
    left_lung_parts = None
    if left_lung_coords is not None:
        left_lung_parts = divide_single_lung(left_lung_coords)
    
    # Return the respective part coordinates for both the right and left lungs
    
    return [right_lung_parts['upper'],left_lung_parts['upper'],right_lung_parts['middle'],left_lung_parts['middle'],right_lung_parts['lower'],left_lung_parts['lower'] ]
        
        
class Context_SiMEXAugmentation:
    def __init__(self,config, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        image_size     = config['data']['resize_size']        
        n_cb           = config['model']['number_of_com_branches']
        self.total_global_views = n_cb+1
        self.total_local_views  = n_cb
        
        self.global_transform = T.Compose([
                                            T.RandomResizedCrop(image_size, scale=(0.3, 1)),
                                            T.RandomRotation(10),
                                            T.RandomHorizontalFlip(),
                                            T.RandomGrayscale(p=0.2),
                                            T.RandomApply([GaussianBlur()], p=0.5),
                                            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                            T.ToTensor(),
                                            T.Normalize(mean=mean, std=std)
                                        ])
        
        self.local_transform = T.Compose([
                                            T.Resize((image_size, image_size)),
                                            T.RandomRotation(10),
                                            T.RandomHorizontalFlip(),
                                            T.RandomGrayscale(p=0.2),
                                            T.RandomApply([GaussianBlur()], p=0.5),
                                            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                            Custom_Distoration(image_size,image_size),
                                            T.ToTensor(),
                                            T.Normalize(mean=mean, std=std)
                                        ])
        
               
    def __call__(self, img, right_lung_box, left_lung_box):
        Global_views = []
        Local_views = []
        
        # Generate global views for each branch
        for _ in range(self.total_global_views):
            global_view = self.global_transform(img)
            Global_views.append(global_view)
            
            
        part_coordinates =  divide_lung_into_parts(right_lung_box, left_lung_box)
        for (x, y, width, height) in part_coordinates:
            local_crop = img.crop((x, y, x + width, y + height))
            local_view = self.local_transform(local_crop)
            Local_views.append(local_view)
            
        return [Global_views, Local_views] 
                
          
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        