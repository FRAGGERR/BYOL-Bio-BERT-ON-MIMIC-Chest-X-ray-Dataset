import torch
from torchvision import transforms
import cv2
import yaml
import pandas as pd
from PIL import ImageFilter
import random
import ast
from PIL import Image, ImageOps
import numpy as np
import copy
import torch.nn.functional as F
from PIL import ImageFile
from torch import Tensor
import skimage.exposure
from numpy.random import default_rng
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img   = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))
    
class Solarize():
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, sample):
        return ImageOps.solarize(sample, self.threshold)
    
    
class RandomApplylungs(torch.nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self,img,right_lung_boxT, left_lung_boxT):
        if self.p < torch.rand(1):
            return img
        for t in self.transforms:
            img = t(img,right_lung_boxT, left_lung_boxT)
        return img
    
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

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,right_lung_boxT, left_lung_boxT):
        for t in self.transforms:
            img = t(img,right_lung_boxT, left_lung_boxT)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
    
class motion_blur:
    def __init__(self, severity=1):
        self.severity = severity
     

    def __call__(self, img, right_lung_box,left_lung_box):
        img_np = np.array(img)

        if all(right_lung_box):
            # Separate the image into right and left lung regions
            right_x, right_y, right_width, right_height = right_lung_box
            right_lung_img = img_np[right_y:right_y + right_height, right_x:right_x + right_width, :]
            # Apply motion blur to right lung
            right_lung_img = self.apply_motion_blur(right_lung_img, severity=self.severity)
            img_np[right_y:right_y + right_height, right_x:right_x + right_width, :] = right_lung_img
            
        if all(left_lung_box):    
            left_x, left_y, left_width, left_height = left_lung_box
            left_lung_img = img_np[left_y:left_y + left_height, left_x:left_x + left_width, :]
            # Apply motion blur to left lung
            left_lung_img = self.apply_motion_blur(left_lung_img, severity=self.severity)
            # Place the modified lung regions back into the original image
            img_np[left_y:left_y + left_height, left_x:left_x + left_width, :] = left_lung_img
            
        if right_lung_box == [0,0,0,0] and left_lung_box == [0,0,0,0]:
            # Apply motion blur to the whole image
            img_np = self.apply_motion_blur(img_np, severity=self.severity)

        return Image.fromarray(img_np)
    
    @staticmethod
    def apply_motion_blur(image, severity=1):
        # Define motion blur kernel size based on severity
        kernel_size = severity * 5

        # Generate motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size

        # Apply motion blur filter to the image
        image_blurred = cv2.filter2D(image, -1, kernel)
        return image_blurred
    
class AnatomicalNoise:
    def __init__(self, severity=1):
        self.severity = severity

    def __call__(self, img, right_lung_box, left_lung_box):
        img_np = np.array(img)
        opacity_color = (183, 183, 183)  # Light grey for opacity

        if all(right_lung_box):
            right_x, right_y, right_width, right_height = right_lung_box
            right_lung_img = img_np[right_y:right_y + right_height, right_x:right_x + right_width, :]

            for _ in range(self.severity):
                radius = np.random.randint(10, min(right_width, right_height) // 2)
                x = np.random.randint(radius, right_width - radius)
                y = np.random.randint(radius, right_height - radius)
                cv2.circle(right_lung_img, (x, y), radius, tuple(opacity_color), -1)

            img_np[right_y:right_y + right_height, right_x:right_x + right_width, :] = right_lung_img

        if all(left_lung_box):
            left_x, left_y, left_width, left_height = left_lung_box
            left_lung_img = img_np[left_y:left_y + left_height, left_x:left_x + left_width, :]

            for _ in range(self.severity * 2):
                radius = np.random.randint(10, min(left_width, left_height) // 2)
                x = np.random.randint(radius, left_width - radius)
                y = np.random.randint(radius, left_height - radius)
                cv2.circle(left_lung_img, (x, y), radius, tuple(opacity_color), -1)

            img_np[left_y:left_y + left_height, left_x:left_x + left_width, :] = left_lung_img

        if right_lung_box == [0, 0, 0, 0] and left_lung_box == [0, 0, 0, 0]:
            for _ in range(self.severity):
                radius = np.random.randint(10, min(img_np.shape[1], img_np.shape[0]) // 2)
                x = np.random.randint(radius, img_np.shape[1] - radius)
                y = np.random.randint(radius, img_np.shape[0] - radius)
                color = np.random.randint(0, 255, size=(3,))
                color = (int(color[0]), int(color[1]), int(color[2]))
                cv2.circle(img_np, (x, y), radius, tuple(color), -1)

        return Image.fromarray(img_np)
    
    
# # Define a custom transformation for simulating pathology markers
class SimulatedPathology:
    def __init__(self, pathology_type='tumor', severity=1):
        self.pathology_type = pathology_type
        self.severity = severity

    def __call__(self, img, right_lung_box,left_lung_box):
        img_np = np.array(img)
        opacity_color = (183, 183, 183)

        if all(right_lung_box): 
            # Generate a distorted round shape for the right lung
            right_lung_x, right_lung_y, right_lung_width, right_lung_height = right_lung_box
            right_lung_height = int(right_lung_height/1.5)
            center_x_r = right_lung_x + right_lung_width // 2
            center_y_r = right_lung_y + right_lung_height // 2
            radius_r = min(right_lung_width, right_lung_height) // 5
            num_points_r = 50
            angle_range_r = np.linspace(0, 2 * np.pi, num_points_r)
            distort_factor_r = 10  # Adjust the distortion factor as needed

            points_x_r = center_x_r + radius_r * np.cos(angle_range_r) + np.random.randint(-distort_factor_r, distort_factor_r, size=num_points_r)
            points_y_r = center_y_r + radius_r * np.sin(angle_range_r) + np.random.randint(-distort_factor_r, distort_factor_r, size=num_points_r)

            # Create a list of points for the contour of the right lung
            contour_points_r = np.array([points_x_r, points_y_r], dtype=np.int32).T
            cv2.fillPoly(img_np[right_lung_y:right_lung_y + right_lung_height, right_lung_x:right_lung_x + right_lung_width, :], [contour_points_r], opacity_color)  # Red color for the distorted shape in the right lung

        if all(left_lung_box):
            # Draw the distorted round shape for the right lung (customize the color)

            # Generate a distorted round shape for the left lung
            left_lung_x, left_lung_y, left_lung_width, left_lung_height = left_lung_box
            left_lung_height = int(left_lung_height/1.5)
            center_x_l = left_lung_x + left_lung_width // 2
            center_y_l = left_lung_y + left_lung_height // 2
            radius_l = min(left_lung_width, left_lung_height) // 5
            num_points_l = 50
            angle_range_l = np.linspace(0, 2 * np.pi, num_points_l)
            distort_factor_l = 10  # Adjust the distortion factor as needed

            points_x_l = center_x_l + radius_l * np.cos(angle_range_l) + np.random.randint(-distort_factor_l, distort_factor_l, size=num_points_l)
            points_y_l = center_y_l + radius_l * np.sin(angle_range_l) + np.random.randint(-distort_factor_l, distort_factor_l, size=num_points_l)

            # Create a list of points for the contour of the left lung
            contour_points_l = np.array([points_x_l, points_y_l], dtype=np.int32).T

            # Draw the distorted round shape for the left lung (customize the color)
            cv2.fillPoly(img_np[left_lung_y:left_lung_y + left_lung_height, left_lung_x:left_lung_x + left_lung_width, :], [contour_points_l], opacity_color)  # Red color for the distorted shape in the left lung
        
        if right_lung_box == [0,0,0,0] and left_lung_box == [0,0,0,0]:
            # Simulate tumors as simple circles (customize the color)
            if self.pathology_type == 'tumor':
                for _ in range(self.severity):
                    radius = np.random.randint(5, 10)
                    x = np.random.randint(radius, img_np.shape[1] - radius)
                    y = np.random.randint(radius, img_np.shape[0] - radius)
                    opacity_color = (183, 183, 183)
                    cv2.circle(img_np, (x, y), radius, opacity_color, -1)

        return Image.fromarray(img_np)
    
def divide_lung_into_parts(right_lung_coords, left_lung_coords, part):
    # Define a function to divide a single lung into parts
    def divide_single_lung(lung_coords):
        # Unpack lung coordinates
        x, y, width, height = lung_coords
        
        # Calculate midline of the lung bounding box
        midline_x = x + width // 2
        
        # Divide the lung into three parts vertically: upper, middle, lower
        upper_y = y
        middle_y = upper_y + height // 3
        lower_y = middle_y + height // 3
        
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
    if part == 'upper':
        return {'right': right_lung_parts['upper'] if right_lung_parts is not None else None,
                'left': left_lung_parts['upper'] if left_lung_parts is not None else None}
    elif part == 'middle':
        return {'right': right_lung_parts['middle'] if right_lung_parts is not None else None,
                'left': left_lung_parts['middle'] if left_lung_parts is not None else None}
    elif part == 'lower':
        return {'right': right_lung_parts['lower'] if right_lung_parts is not None else None,
                'left': left_lung_parts['lower'] if left_lung_parts is not None else None}
    else:
        raise ValueError("Invalid part specified. Valid parts are 'upper', 'middle', or 'lower'.")



def get_nodule_center(right_lung_coords,left_lung_coords):
    # getting the bounding coordinates for the left and the right lung
    
    if left_lung_coords is not None:
        x_left, y_left, width_left, height_left = left_lung_coords
    if right_lung_coords is not None:
        x_right, y_right, width_right, height_right = right_lung_coords
    
    choice_list = ['left', 'right']
    choice = random.choice(choice_list)
    
    #we try to create a window around the center point of the lung coordinates
    if choice == 'left':
        mean_x = x_left + width_left//2
        mean_y = y_left + height_left//2
        #the values 20 and 75 are empirical value: taking larger values will make the window size larger
        #try to keep the values for x-axis and y axis higher as we want the window to be a rectangular size
        x = np.random.randint(mean_x - 20, mean_x + 20)
        y = np.random.randint(mean_y - 75, mean_y+75)
        center = (x,y)
        
    elif choice == 'right':
        mean_x = x_right + width_right//2
        mean_y = y_right + height_right//2
        x = np.random.randint(mean_x - 20, mean_x + 20)
        y = np.random.randint(mean_y - 75, mean_y+75)
        center = (x,y)
        
    return center, choice

def generate_nodules_v2(image, center_list, radius_list, num, seed, threshold, brightness, locate = False):
    '''
        nodule_center : location where the circle mask is created
        nodule_radius : radius of the circle
        threshold : randomizes the structure of the texture (higher values make the size of indivisual nodule bigger)
        brightness : higher value makes the nodule more visible
        locate : if true, creates a red circle around the nodule
    '''
    height, width = image.shape[:2]
    rng = default_rng(seed=seed)
    
    #creating the initial texture
    noise   = rng.integers(0, 255, (height,width), np.uint8, True)
    blur    = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)
    thresh  = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask1  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #creating a new mask to apply to the previous mask
    
    mask2 = np.zeros_like(image[:, :,0])
    for i in range(num):
        image_color = (225,225,225)
        cv2.circle(mask2, center_list[i], radius_list[i], image_color, -1)
        mask = cv2.bitwise_and(mask1, mask1, mask = mask2)
    nodule_image = cv2.bitwise_and(image, image, mask = mask)

    mask_image = cv2.merge([mask,mask,mask])
    mask_image = cv2.threshold(mask_image, 230, brightness, cv2.THRESH_BINARY)[1]
    result     = cv2.addWeighted(image,0.95, mask_image, 0.2, 0)
    
    if locate:
        for i in range(num):
            cv2.circle(result, center_list[i], radius_list[i]+30, (0,0,255), 2)
    return result    

class lung_nodule_generate:
    def __init__(self, severity=1, th = [125,145], bt = [110, 140], rd = [90, 120],parts =False ):
        self.severity = severity
        self.seed = np.random.randint(1,100)
        self.threshold = np.random.randint(th[0],th[1])
        self.brightness = np.random.randint(bt[0],bt[1])
        self.nodule_radius = np.random.randint(rd[0],rd[1]) 
        self.parts = parts

    def __call__(self, img, right_lung_box,left_lung_box):
        image = np.array(img)
        center_list = []
        radius_list = []
        
        if self.parts == True:
            choice_list = ['lower', 'upper', 'middle']
            choice = random.choice(choice_list)
            part_coordinates =  divide_lung_into_parts(right_lung_box, left_lung_box, choice)
            right_lung_box = part_coordinates['right']
            left_lung_box  = part_coordinates['left']
        
        for i in range(self.severity):
            nodule_center, choice = get_nodule_center(right_lung_box,left_lung_box)
            center_list.append(nodule_center)
            radius_list.append(self.nodule_radius)
        
        nodule_image = generate_nodules_v2(image,center_list,radius_list,self.severity,self.seed,self.threshold,self.brightness)

        return Image.fromarray(nodule_image)
    
    
def resize_bbox(orig_image, resize_image, right_bbox,left_bbox):
    
    orig_image_size   = orig_image.size
    resize_image_size = resize_image.size()
    
    width_scale = resize_image_size[1] / orig_image_size[0]
    height_scale = resize_image_size[2] / orig_image_size[1]
    
    new_right_bbox = [
                int(right_bbox[0] * width_scale),    # New x-coordinate of the top-left corner
                int(right_bbox[1] * height_scale),   # New y-coordinate of the top-left corner
                int(right_bbox[2] * width_scale),    # New x-coordinate of the bottom-right corner
                int(right_bbox[3] * height_scale)    # New y-coordinate of the bottom-right corner
            ]
    
    new_left_bbox = [
            int(left_bbox[0] * width_scale),    # New x-coordinate of the top-left corner
            int(left_bbox[1] * height_scale),   # New y-coordinate of the top-left corner
            int(left_bbox[2] * width_scale),    # New x-coordinate of the bottom-right corner
            int(left_bbox[3] * height_scale)    # New y-coordinate of the bottom-right corner
        ]
    
    return new_right_bbox, new_left_bbox

class random_lung_masking:
    def __init__(self, mask_percentage):
        self.mask_percentage = mask_percentage
        
    def __call__(self, image, rbbox, lbbox):
        image_size = (224, 224)
        to_tensor = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor()])

        resize_image = to_tensor(image)
        rbbox, lbbox = resize_bbox(image, resize_image, rbbox, lbbox)

        img_height = resize_image.shape[1]
        img_width = resize_image.shape[2]

        mask = torch.ones((3, img_height, img_width), dtype=torch.float32)

        if rbbox is not None:
            xr_min, yr_min, widthr, heightr = rbbox
            rbbox_width = widthr
            rbbox_height = heightr
            patch_size = 8
            if rbbox_width > patch_size and rbbox_height > patch_size:
                num_patches = int(self.mask_percentage * (rbbox_height * rbbox_width) / (patch_size * patch_size))
                for _ in range(num_patches):
                    x = np.random.randint(xr_min, xr_min + rbbox_width - patch_size)
                    y = np.random.randint(yr_min, yr_min + rbbox_height - patch_size)
                    mask[:, y:y+patch_size, x:x+patch_size] = 0.0

        if lbbox is not None:
            xl_min, yl_min, widthl, heightl = lbbox
            lbbox_width = widthl
            lbbox_height = heightl
            patch_size = 8
            if lbbox_width > patch_size and lbbox_height > patch_size:
                num_patches = int(self.mask_percentage * (lbbox_height * lbbox_width) / (patch_size * patch_size))
                for _ in range(num_patches):
                    x = np.random.randint(xl_min, xl_min + lbbox_width - patch_size)
                    y = np.random.randint(yl_min, yl_min + lbbox_height - patch_size)
                    mask[:, y:y+patch_size, x:x+patch_size] = 0.0

        masked_image = resize_image * mask

        return masked_image

    
def get_genric_transform(gb_p=0.5,cj_p=0.5,rg_p=0.5,so_p = 0.5):
                                   
    t_list = []
    color_jitter = transforms.ColorJitter(0.6, 0.6, 0.6, 0.2)
    t_list =[
            transforms.RandomApply([color_jitter], p=1.0),
            transforms.RandomApply([GaussianBlur(kernel_size=23)], p=gb_p),
            transforms.RandomApply([transforms.RandomGrayscale(p=0.5)],p =rg_p),
            transforms.RandomApply([Solarize()], p=so_p),
                             ]
    transform = transforms.Compose(t_list)
    return transform

def get_domain_transform(mb_p=0.5, an_p=0.5, sp_p=0.5,ln_p1=0.5,ln_p2=0.5):
    t_list = [
            RandomApplylungs([motion_blur(severity=8)],p=mb_p),  
            RandomApplylungs([AnatomicalNoise(severity=1)],p=an_p),
            RandomApplylungs([SimulatedPathology('tumor', 2)],p=sp_p),
            RandomApplylungs([lung_nodule_generate(severity=1,th = [120,135], bt = [130, 150], rd = [20, 30],parts=False )],p=ln_p1),
            RandomApplylungs([lung_nodule_generate(severity=80,th = [120,135], bt = [130, 150], rd = [20, 30],parts=True)],p=ln_p2),
            ]
    transform = Compose(t_list)
    return transform

def get_spatial_transform(image_size, hf_p=0.5,rf_p=0.5):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t_list = []
    t_list =[
            transforms.RandomResizedCrop(image_size),
            Custom_Distoration(224,224),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=hf_p),
            transforms.RandomApply([transforms.RandomAffine(degrees=0,translate=(0.2, 0.2))],p=rf_p),
            transforms.ToTensor(),
#             normalize
                             ]
    transform = transforms.Compose(t_list)
    return transform

def get_mask_transformtion(mask_percentage):
    t_list = [random_lung_masking(mask_percentage=mask_percentage)]
    transform = Compose(t_list)
    return transform


class DomainAUgmentations:
    def __init__(self,config ):  #,
        image_size   = config['data']['resize_size']
        
        self.genric_transform1   = get_genric_transform()
        self.domain_transform1   = get_domain_transform()
        self.spatial_transform1  = get_spatial_transform(image_size)
        
        self.genric_transform2   = get_genric_transform()
        self.domain_transform2   = get_domain_transform()
        self.spatial_transform2  = get_spatial_transform(image_size)
        
        self.masked_transform   = get_mask_transformtion(mask_percentage=1.0)
        
        self.to_tensor = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor()])

         
        
    def __call__(self, x, right_lung_box, left_lung_box):
        x1 = self.spatial_transform1(self.domain_transform1(self.genric_transform1(x),right_lung_box,left_lung_box))
        x2 = self.spatial_transform2(self.domain_transform2(self.genric_transform2(x),right_lung_box,left_lung_box))
        x3 = self.masked_transform(x,right_lung_box,left_lung_box)
        orig = self.to_tensor(x)
        return [orig, x1, x2, x3]