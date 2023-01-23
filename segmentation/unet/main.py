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
from dataset import MonitorDataset 
from utils import *
from datetime import datetime
import segmentation_models_pytorch as smp



class CONFIG:
    NUM_EPOCHS = 10
    DEVICE = torch.device('cuda')
    lossFunc = torch.nn.BCEWithLogitsLoss()
    INIT_LR = 0.001
    labels_file = './labels.csv'
    images_folder = './images'
    masks_folder = './masks'
    train_batch = 4
    test_batch = 4
    train_steps = 0
    test_steps = 0

    encoder_weights = None
    encoder_depth = 5
    encoder_name = 'resnet34'
    decoder_channels = [256, 128, 64, 32, 16]

    description = 'none'
    output_path = 'unet_model'
    
if __name__ == '__main__':

    config = CONFIG()

    dataset = MonitorDataset(
                labels_file=config.labels_file,
                images_folder = config.images_folder,
                masks_folder = config.masks_folder)
    L = len(dataset)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [])
    config.train_steps = len(train_ds) // config.train_batch
    config.test_steps = len(val_ds) // config.test_batch
    train_dl = DataLoader(train_ds, batch_size=config.train_batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=config.test_batch, shuffle=True, num_workers=0)

    unet = smp.Unet(encoder_weights = config.encoder_weights,
                    encoder_depth = config.encoder_depth,
                    encoder_name = config.encoder_name,
                    decoder_channels = config.decoder_channels, 
                    classes=2).cuda()

    opt = torch.optim.Adam(unet.parameters(), lr=config.INIT_LR)

    H = train(unet, train_dl, val_dl, config)
    save_dict = {'state_dict': unet.state_dict(), 
                'iou_score': avgiou, 
                'description': config.description}
    output_path += '_'+str(datetime.now())+'.pth'
    torch.save(save_dict, output_path)

