import torch
import torch.utils.data
import PIL
import os
import numpy as np
from glob import glob

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class customData(torch.utils.data.Dataset):
    def __init__(self, young_path, old_path, data_transforms=None, loader=My_loader):
        self.young_files = glob(young_path+'/*.jpg')
        self.old_files = glob(old_path+'/*.jpg')
        self.young_len = len(self.young_files)
        self.old_len = len(self.old_files)
        self.file_len = max(self.young_len, self.old_len)
        self.loader = loader
        self.data_transforms = data_transforms

    def __len__(self):
        return self.file_len

    def __getitem__(self, item):
        young_name = self.young_files[item % self.young_len]
        old_name = self.old_files[item % self.old_len]
        
        young_img = self.loader(young_name)
        old_img = self.loader(old_name)

        if self.data_transforms is not None:
            young_img = self.data_transforms(young_img)
            old_img = self.data_transforms(old_img)
        
        return young_img, old_img