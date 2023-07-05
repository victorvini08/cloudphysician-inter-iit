
import yaml
import cv2
import os
import time
import numpy as np
import warnings
import torch
from torchvision.ops import nms
from ocr import ocr_model
from segment import Segment
from ISD_SSD.ssd import build_ssd
from ISD_SSD.ssd_inference import run_ssd
from hr_digitization_final import plot_digitizer
from post_processing import post_process_boxes,nms
from feature_match_ocr_copy import perform_ocr

warnings.filterwarnings("ignore")

#LOAD CONFIG FILE
with open('./config.yaml') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)

#LOAD SEGMENTATION MODEL
seg = Segment(config)

#LOAD OBJECT DETECTION MODEL
num_classes = 9 + 1


trained_model = config['trained_model']
if '300' in trained_model:
  net = (build_ssd('test', 300, num_classes)).cpu() # initialize SSD
else:
  net = (build_ssd('test', 512, num_classes)).cpu() # initialize SSD
net.load_state_dict(torch.load(trained_model,map_location=torch.device('cpu')))

#LOAD OCR MODEL
ocr = ocr_model(config)

def inference(image_path:str):
  '''
  Function responsible for inference.
  Args: 
    image_path: str, path to image file. eg. "input/aveksha_micu_mon--209_2023_1_17_12_0_34.jpeg"
  Returns:
    result: dict, final output dictionary. eg. {"HR":"80", "SPO2":"98", "RR":"15", "SBP":"126", "DBP":"86"}
  '''
  result = {}
  

  image = cv2.imread(image_path)
  if image is None:
    raise FileNotFoundError("Image path does not exist! Please change path.")
    exit()

  try:
    segmented_img = seg.run(image)
    cv2.imwrite("/content/seg_img.jpg",segmented_img)
  except:
    raise RuntimeError('Screen Segmentation Failed')
  start = time.time()
  obj_det_output = run_ssd(config,segmented_img,net)
  print('OD Time: ', time.time()-start)
  #[{'bbox': [1,2,3,4], 'label_name': 'HR', 'score':}]
  post_processed_boxes = post_process_boxes(obj_det_output,segmented_img,int(config['extra_size_bbox']))
  
  list_boxes = []
  list_labels = []
  list_scores = []
  for i in post_processed_boxes:
    list_boxes.append(i["bbox"])
    list_labels.append(i['label_name'])
    list_scores.append(i['score'])

  list_boxes, list_labels, _ = nms(list_boxes, list_labels, list_scores, 0.7)

  out = ocr.draw_bbox(img = segmented_img,boxes = list_boxes,labels = list_labels, scores = list_scores)

  cv2.imwrite("/content/bbox_output.jpg",out)
  if 'HR_W' in list_labels:
    idx =  list_labels.index('HR_W')
    hrw_bbox = list_boxes[idx]
    print("Digitizing HR")
    plot_digitizer(segmented_img, hrw_bbox)

  list_labels_nographs = []
  list_boxes_nographs = []

  for label, box in zip(list_labels, list_boxes):
    if label[-1] != 'W':
      list_labels_nographs.append(label)
      list_boxes_nographs.append(box)

  if config['ocr']['method'] == 'feature_matching':
    print('Using Feature Matching Method for OCR')
    ans = perform_ocr(segmented_img, list_boxes_nographs, list_labels_nographs)
  else:
    ans = ocr.run_bbox(segmented_img, list_boxes_nographs, list_labels_nographs)
    print("finished ocr!")

  ### put your code here

  return ans
  
if __name__ == '__main__':
  imgpath = '/content/drive/MyDrive/data/Unlabelled_Dataset/prashant_icu_mon--9_2023_1_1_7_18_26.jpeg'
  print('Image path: ', imgpath.split('/')[-1])
  start = time.time()
  ans = inference(imgpath)
  print("total time: ",time.time()-start)
  print(ans)
