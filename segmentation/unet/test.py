import argparse
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
import segmentation_models_pytorch as smp
from dataset import *
from imutils import perspective


parser = argparse.ArgumentParser()

parser.add_argument('--model-path', default = None, type = str)
parser.add_argument('--image-folder', default = None, type = str)
parser.add_argument('--output-dir', default = None, type = str)
parser.add_argument('--padding', default = 10, type = int)

if __name__ == '__main__':
    

    args = parser.parse_args()
    args.model_path = 'unet_model_2.pth'
    args.image_folder = '/home/raj/Desktop/Unlabelled Dataset'
    args.output_dir = './unlabelled-seg'
    print('Model Path: ', args.model_path)
    print('Source Directory: ', args.image_folder)
    print('Output Directory: ', args.output_dir)

    model = torch.load(args.model_path, 'cpu')

    unet = smp.Unet(encoder_weights = model['encoder_weights'],
                    encoder_depth = model['encoder_depth'],
                    encoder_name = model['encoder_name'],
                    decoder_channels = model['decoder_channels'], 
                    classes=2).cuda()


    unet.load_state_dict(model['state_dict'])
    dataset = MonitorInferenceDataset(images_folder = args.image_folder)
    dataloader = DataLoader(dataset,  batch_size = 1, shuffle=True, num_workers=0)
    if not os.path.isdir(args.output_dir):
      os.mkdir(args.output_dir)
    print('Number of images: ', len(dataset))
    for filenames, img in tqdm(dataloader):
        # img = img.cuda()
        filename = filenames[0]
        image = cv2.imread(os.path.join(args.image_folder, filename))
        h,w,c = image.shape
        pred = unet(img.cuda())[0].detach().cpu().numpy()
        pred = np.argmax(pred, axis=0)
        pred = np.reshape(pred, (pred.shape[0], pred.shape[1], 1)).astype(np.float32)
        pred = cv2.resize(pred, (w, h))
        cv2.imwrite('/home/raj/cloudphy/monitor-segmentation-master/unet/mask/'+filename,pred*255)

        contours, hierarchy = cv2.findContours(pred.astype(np.uint8), 
                                                cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE)
        max_area_idx = 0
        max_area = 0

        for i, contour in enumerate(contours):
          area = cv2.contourArea(contour)
          if area > max_area:
            max_area = area
            max_area_idx = i

        box = cv2.minAreaRect(contours[max_area_idx])
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int32")
        box = perspective.order_points(box).astype(np.int32)

        box[0] -= 10
        box[1][0] += 10
        box[1][1] -= 10
        box[2] += 10
        box[3][0] -= 10
        box[3][1] += 10
        mask = np.zeros(pred.shape)
        mask = cv2.fillPoly(mask, pts = [box] , color =(1,1,1))
        
        seg = np.zeros(image.shape)
        image[:,:,0] = image[:,:,0]*mask
        image[:,:,1] = image[:,:,1]*mask
        image[:,:,2] = image[:,:,2]*mask

        target = np.array([[0,0], [w-1,0], [w-1,h-1], [0, h-1]])

        matrix = cv2.getPerspectiveTransform(box.astype(np.float32), target.astype(np.float32))
        image = cv2.warpPerspective(image, matrix, (w,h))

        cv2.imwrite(os.path.join(args.output_dir, filename), image)

        









    

 