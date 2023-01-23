from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image, ImageOps
import torch
import pandas as pd
import cv2
import time
from skimage import io
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as tt



class MonitorDataset(Dataset):
    def __init__(self, 
                 labels_file, 
                 images_folder, 
                 masks_folder, 
                 transform = tt.Compose([tt.Resize((192, 320)), tt.ToTensor()])):
        self.labels = pd.read_csv(labels_file)
        self.filenames = os.listdir(images_folder)
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.transform = transform
        self.b_rects = {}
        for i in range(len(self.labels)):
            file = self.labels.iloc[i]['image_name']
            points = self.labels.iloc[i]['points'][1:-1].replace(' ', '').split(',')
            points = [float(p) for p in points]
            points = np.reshape(points, (4,2)).astype(np.int32)
            self.b_rects[file] = points
            
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        file = self.filenames[idx]
        path = os.path.join(self.images_folder, file)
        img = Image.open(path)
        
        path = os.path.join(self.masks_folder, file)
        mask = Image.open(path)
        mask = ImageOps.grayscale(mask)
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        points = self.b_rects[file]
        return img, mask, points





class MonitorInferenceDataset(Dataset):
    def __init__(self, 
                 images_folder, 
                 transform = tt.Compose([tt.Resize((192, 320)), tt.ToTensor()])):
        self.filenames = os.listdir(images_folder)
        self.images_folder = images_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        
        filename = self.filenames[idx]
        path = os.path.join(self.images_folder, filename)
        img = Image.open(path)
        
        if self.transform:
            img = self.transform(img)
        return filename, img