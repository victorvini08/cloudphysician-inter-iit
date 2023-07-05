from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from .data import *
from PIL import Image
from .data import COCO_ROOT, MONITOR_CLASSES as labelmap
import torch.utils.data as data
import math
import time
import cv2
import numpy as np
import warnings
import pandas as pd
from torch import autocast

def unlabelled_data():
  unlabelled_data_list = []
  root_unlabelled_path =  "/content/drive/MyDrive/coco-with-graphs-data/unlabelled2017/"
  for path in os.listdir(root_unlabelled_path):
    path = os.path.join(root_unlabelled_path,path)
    unlabelled_data_list.append(path)

  return unlabelled_data_list

@torch.no_grad()
def test_net(net, cuda, img, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    # df = pd.read_csv(gt_labels)
    mAP = []
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    if cuda:
        x = x.cuda()
    y = net(x)      # forward pass
    detections = y.data

    # scale each detection back up to the image
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    pred_num = 0
    outputs = []
    
    for i in range(detections.size(1)):
        j = 0
        color = (36,255,12)
        list_of_boxes = []

        while detections[0, i, j, 0] >= 0.3:
            score = detections[0, i, j, 0]
            label_name = labelmap[i-1]

            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3])

            
            label_name = label_name 
            
            pt = pt.astype(np.uint16)
            d = {"score":score,"label_name":label_name,"pt":pt}
            list_of_boxes.append(d)

            pred_num += 1
            j += 1
        newlist = sorted(list_of_boxes, key=lambda d: d['score'],reverse=True) 
        if len(newlist)>0:
          box = newlist[0]
          pt = box['pt']
          label_name = box["label_name"]
          score = box['score']
          
          xmin = pt[0]
          ymin = pt[1]
          xmax = pt[2]
          ymax = pt[3]
          bbox = (xmin,ymin,xmax,ymax)
          final_per_class = {'bbox':bbox,'label_name':label_name,'score':score.item()}
          outputs.append(final_per_class)
    return outputs




def run_ssd(config,image,net):

    trained_model = config['trained_model']
    visual_threshold = config['visual_threshold']
    cuda_flag = config['cuda_flag']
    if cuda_flag and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    net.eval()
    # print('Finished loading model!')
    # load data
    #testset = COCODetection(root=dataset_root)
    #print(testset)
    #unlabelled_set = unlabelled_data()
    # testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if cuda_flag:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    outputs = test_net(net, cuda_flag, image,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=visual_threshold)
    return outputs
    
