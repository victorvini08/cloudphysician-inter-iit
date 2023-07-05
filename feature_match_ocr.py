import numpy as np
import cv2
import time
import os
from matplotlib import pyplot as plt
from torch.jit.annotations import parse_type_line

templates = {}
method = 'contour' # mser, contour

template_root = './template_crop'

for folder in os.listdir(template_root):
  for f in os.listdir(os.path.join(template_root, folder)):
    if '.png' in f:
      image_path = os.path.join(template_root, folder, f)
      img = cv2.imread(image_path, 0)
      templates[image_path] = img/255

def iou_score(query, template):
  h,w = template.shape
  query = cv2.resize(query, (w,h))
  intersection = np.sum(query*template)
  union = np.sum(((query+template)>0)*1)
  return (intersection/union)


def mser(cv_image):
    mser = cv2.MSER_create(min_area=800, max_area=4000, edge_blur_size=0)
    regions, _ = mser.detectRegions(cv_image)
    boxes = []
    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def match_iou(digit):
  max_score = 0
  c = None
  for k in templates.keys():
    n = templates[k]
    iou = iou_score(digit, n)
    if iou > max_score:
      max_score = iou
      c = k.split('/')[-1][0]
  
  return c



def image_to_string(j, img, invert = False):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  h,w = img.shape
  img = cv2.resize(img, (int(100*w/h), 100))
  ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  kernel = np.ones((3, 3), np.uint8)
  img = cv2.erode(img, kernel) 
  #img = cv2.erode(img, kernel) 
  if invert:
    img = np.abs((img/255) - 1)*255
    img = img.astype(np.uint8)

  if method == 'mser':
    boxes = mser(img)
    if len(boxes) == 1:
      return ''
    values = {}
    for box in boxes:
      xmin = box[0]
      ymin = box[1]
      xmax = box[2]
      ymax = box[3]

      digit = img[ymin:ymax, xmin:xmax]/255
      c = match_iou(digit)
      values[xmin] = c
  
  elif method == 'contour':

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areas = np.array([cv2.contourArea(c) for c in contours])
    #print(areas)
    values = {}
    for i,c in enumerate(contours):
      if cv2.contourArea(c) > 1000:
        x,y,w,h = cv2.boundingRect(c)
        digit = img[y:y+h, x:x+w]/255
        c = match_iou(digit)
        values[x] = c
  
  ordered = sorted(values)
  string = ''
  for o in ordered:
    string += values[o]
  

  return string



  # contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  # areas = np.array([cv2.contourArea(c) for c in contours])
  # #print(areas)
  # values = {}
  # for i,c in enumerate(contours):
  #   if cv2.contourArea(c) > 1000:
  #     x,y,w,h = cv2.boundingRect(c)
  #     digit = img[y:y+h, x:x+w]/255
  #     cv2.imwrite('/content/crop'+str(j)+str(i)+'.jpeg', digit*255)
  #     max_score = 0
  #     c = None
  #     for k in templates.keys():
  #       n = templates[k]
  #       iou = iou_score(digit, n)
  #       #print(k, iou)
  #       if iou > max_score:
  #         max_score = iou
  #         c = k
    
  #     values[x] = c

  # order = sorted(values)
  # string = ''
  # for o in order:
  #   string += values[o]

  # end = time.time()
  # #print('Time taken for OCR (feature matching): ', end-start)
  # return string


def perform_ocr(image, list_boxes, list_labels):
  result = {}
  s = 0
  for i,box in enumerate(list_boxes):
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    label = list_labels[i]
    #print(xmin, ymin, xmax, ymax)
    crop = image[ymin:ymax, xmin:xmax, :]
    #cv2.imwrite('/content/CROP_'+str(i)+'.jpeg', crop)
    string= image_to_string(i, crop)
    if len(string) < 2:
      string= image_to_string(i, crop, True)
  
    result[label] = string
  return result
  

