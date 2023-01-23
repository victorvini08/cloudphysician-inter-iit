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


def one_hot(targets):    
    targets_extend=targets.clone()# convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), 2, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend.type(torch.int64), 1) 
    return one_hot

def iou_score(preds, masks):
    #print(preds.shape, masks.shape)
    union = ((preds+masks)>0)*1
    intersection = (preds*masks)
    u = torch.sum(union, axis=(1,2))
    i = torch.sum(intersection, axis=(1,2))
    iou = torch.mean(i/u)
    return iou


def train(unet, train_dl, val_dl, config):
    print("[INFO] training the network...")
    startTime = time.time()
    H = {"train_loss": [], "test_loss": [], "iou_score": []}


    for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        for (x, y, points) in tqdm(train_dl):
            # send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # perform a forward pass and calculate the training loss
            pred = unet(x)
            y = one_hot(y)
            loss = config.lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            config.opt.zero_grad()
            loss.backward()
            config.opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            totaliou = 0
            for (x, y, points) in tqdm(val_dl):
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                preds = unet(x)
                masks = torch.reshape(y, (config.train_batch, 192, 320))
                y = one_hot(y)
                totalTestLoss += config.lossFunc(preds, y)
                preds = torch.argmax(preds, axis=1)
                iou = iou_score(preds, masks)
                totaliou += iou_score(preds, masks)
            
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / config.train_steps
        avgTestLoss = totalTestLoss / config.test_steps
        avgiou = totaliou/testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H['iou_score'].append(avgiou.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}, IOU Score: {:.4f}".format(
            avgTrainLoss, avgTestLoss, avgiou))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    return H
