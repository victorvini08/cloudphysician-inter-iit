import numpy as np
import cv2

def post_process_boxes(bounding_boxes,image,extra_size):
  '''
  bounding_boxes [ {'bbox': [] , 'label_name' : [], 'score': } ]

  '''
  image_width = image.shape[1]
  image_height = image.shape[0]
  for bbox in bounding_boxes:
    box = list(bbox['bbox'])
    box[0]-=extra_size
    box[1]-=extra_size
    box[2]+=extra_size
    box[3]+=extra_size
    if box[0]<0 or box[0]>=image_width:
      box[0] = 0
    if box[1]<0:
      box[1] = 0
    if box[2]>=image_width:
      box[2] = image_width-1
    if box[3]>=image_height:
      box[3] = image_height-1
    bbox['bbox'] = box

  return bounding_boxes

def nms(bounding_boxes,labels,confidence_score,threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_labels = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_labels.append(labels[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_labels, picked_score

def post_processing_ocr(ans):
  for l in list(ans):
    ans[l] = str(ans[l])
  old_ans = ans.copy()
  for l in list(ans):
    # print("L",l)
    char = ans[l]
    for c in char:
      if c.isalpha():
        ans.pop(l)
        break
  for l in list(ans):
    value = ans[l]
    if '(' in value or ')' in value:
      if l != 'MAP':
        if 'MAP' in list(ans):
          #swap
          t = ans['MAP']
          ans['MAP'] = value
          ans[l] = t
        else:
          ans.pop(l)
          ans['MAP'] = value
  if 'MAP' in list(ans):
    ans['MAP'] = ans['MAP'].replace('(', '')
    ans['MAP'] = ans['MAP'].replace(')', '')
  SBP = 'SBP' in list(ans)
  DBP = 'DBP' in list(ans)
  MAP = 'MAP' in list(ans)
  if SBP:
    if '/' in ans['SBP']:
      ans['SBP'] = ans['SBP'].split('/')[0]
  if DBP:
    if '/' in ans['DBP']:
      ans['DBP'] = ans['DBP'].split('/')[-1]
    ans['DBP'] = ans['DBP'].replace('(', '')
    ans['DBP'] = ans['DBP'].replace(')', '')
  if SBP:
    SBP = SBP and (int(float(ans['SBP'])) < 300)
  if DBP:
    DBP = DBP and (int(float(ans['DBP'])) < 300)
  if MAP:
    MAP = MAP and (int(float(ans['MAP'])) < 300)
  # if SBP and DBP:
  #   if float(ans['SBP']) < float(ans['DBP']):
  #     t = ans['SBP']
  #     ans['SBP'] = ans['DBP']
  #     ans['DBP'] = t
  if 'SBP' in list(ans):
    # print("CHANGING SBP --> ", ans['SBP'])
    ans['SBP'] = ans['SBP'].replace('/', '')
    # print("New SBP ", ans['SBP'])
  if 'DBP' in list(ans):
    ans['DBP'] = ans['DBP'].replace('/', '')
  if SBP and DBP and not MAP:
    ans['MAP'] = str(int(int(float(ans['DBP'])) + (int(float(ans['SBP'])) - int(float(ans['DBP'])))/3))
  if SBP and MAP and not DBP:
    ans['DBP'] = str(int(1.5*int(float(ans['MAP'])) - 0.5*int(float(ans['SBP']))))
  if MAP and DBP and not SBP:
    ans['SBP'] = str(int(3*float(ans['MAP']) - 2*int(float(ans['DBP']))))
  # if 'SPO2' not in list(ans):
  #   ans['SPO2'] = '100'
  # if 'HR' not in list(ans):
  #   ans['HR'] = '72'
  # if 'RR' not in list(ans):
  #   ans['RR'] = '18'
  for l in list(ans):
    value = ans[l]
    remove = []
    for c in value:
      if not (c.isnumeric() or c=='.'):
        remove.append(c)
    for r in remove:
      ans[l] = ans[l].replace(r, '')
  for l in list(ans):
    if(ans[l]==''):
      del ans[l]
  # if 'RR' not in list(ans):
  #   ans['RR'] = '18'
  for l in list(ans):
    ans[l] = float(ans[l])
  if 'SPO2' in list(ans) and int(ans['SPO2']) > 100:
    ans['SPO2'] = 100
  if 'HR' in list(ans) and int(ans['HR']) > 200:
    ans['HR'] = 72
  if 'RR' in list(ans) and type(ans['RR']) is not str:
    ans['RR'] = str(int(ans['RR']))
  if 'RR' in list(ans) and int(ans['RR']) > 60:
    if len(ans['RR']) > 2:
      mindif = 100000
      rr = 0
      L = len(ans['RR'])
      for i in range(L-1):
        rr_ = ans['RR'][i] + ans['RR'][i+1]
        dif = abs(18 - int(rr_))
        if dif < mindif:
          mindif = dif
          rr = rr_
      ans['RR'] = rr
    ans['RR'] = 18
  for l in list(ans):
    ans[l] = int(ans[l])
  return ans