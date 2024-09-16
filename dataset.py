import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AppleOrangeData(Dataset):
    
    def __init__(self, root_dir:str, transform:transforms.Compose=None):
        
        apple_dir = os.path.join(root_dir, 'apple')
        orange_dir = os.path.join(root_dir, 'orange')
        self.apple_images = list(os.listdir(apple_dir))
        self.orange_images = list(os.listdir(orange_dir))
        
        for i in range(len(self.apple_images)):
            self.apple_images[i] = os.path.join(apple_dir, self.apple_images[i])
        for i in range(len(self.orange_images)):
            self.orange_images[i] = os.path.join(orange_dir, self.orange_images[i])    
        
        self.apple_len = len(self.apple_images)
        self.orange_len = len(self.orange_images)
        self.data_len = max(self.apple_len, self.orange_len)
        self.transform = transform
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        apple_path = self.apple_images[idx % self.apple_len]
        orange_path = self.orange_images[idx % self.orange_len]
        
        apple_img = Image.open(apple_path)
        orange_img = Image.open(orange_path)
        
        if self.transform:
            apple_img = self.transform(apple_img)
            orange_img = self.transform(orange_img)
        
        return {
            'Apple' : apple_img,
            'Orange' : orange_img
        }