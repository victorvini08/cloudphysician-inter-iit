import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_digitizer(img, bbox):
    x_min,y_min,x_max,y_max = bbox
    img = img[y_min:y_max,x_min:x_max]
    #cv2.imwrite('/content/hr_graph_original_1.png',img.astype(np.uint8))
    image=img[img.shape[0]//15:,img.shape[1]//10:(img.shape[1]*9)//10]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    output = cv2.connectedComponentsWithStats(img, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(img.shape, dtype="uint8")
    for i in range(1, numLabels):
      x = stats[i, cv2.CC_STAT_LEFT]
      y = stats[i, cv2.CC_STAT_TOP]
      w = stats[i, cv2.CC_STAT_WIDTH]
      h = stats[i, cv2.CC_STAT_HEIGHT]
      area = stats[i, cv2.CC_STAT_AREA]
      keepArea = area > 200
      if keepArea:
        componentMask = (labels == i).astype("uint8") * 255
        mask = cv2.bitwise_or(mask, componentMask)
    img=255-mask
    #cv2.imwrite('/content/hr_graph_thresh_1.png',img.astype(np.uint8))
    pixel_from_top=[]
    from_list=[]
    for i in range (0,img.shape[1]):
        id=0
        for j in range (img.shape[0]):
            if img[j][i]==0:
                pixel_from_top.append(j)
                id=1
                if len(from_list)!=0:
                  for pixel in from_list:
                    pixel_from_top.insert(pixel,j)
                  from_list=[]
                break
        if id==0:
          if len(pixel_from_top)==0:
            from_list.append(i)
          else:
            pixel_from_top.append(pixel_from_top[i-1])
    pixel_from_bottom=[]
    for i in range (0,img.shape[1]):
        pixel_from_bottom.append(img.shape[0]-pixel_from_top[i])

    dpi=80
    f = plt.figure()
    f.set_figwidth(img.shape[1]/float(dpi))
    f.set_figheight(img.shape[0]/float(dpi))
    plt.plot(pixel_from_bottom)
    plt.show()
    plt.savefig('/content/hr_graph_plt_final_1.png')
    return pixel_from_bottom