import os
import ast
import string
import cv2
import torch
import math
import numpy as np
from scipy import ndimage
import time
import keras_ocr
import tf2onnx
import onnxruntime as rt
# from paddleocr import PaddleOCR
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class ocr_model:
    def __init__(self, config):
        self.config = config
        # self.paddleocr_recog = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        #self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
        #self.trocr_model = (VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')).cpu()
        # alphabets = string.digits + "()/" #+ string.ascii_lowercase
        # self.keras_ocr_model = keras_ocr.recognition.Recognizer(weights=None, alphabet=alphabets)
        # self.keras_ocr_model.model.load_weights(self.config["ocr"]["kerasocr_weights_path"])
        # providers = ['CPUExecutionProvider']
        # self.kerasocr_onnx = rt.InferenceSession(self.config["ocr"]["kerasocr_onnx_path"], providers=providers)

    def run_bbox(self, image, bboxes, classes, model):
        h, w, _ = image.shape
        # bboxes = [i["bbox"] for i in bboxes_dict]
        # classes = [i["label_name"] for i in bboxes_dict]
        
        #bboxes = self.preprocess_bbox(bboxes, h, w)
        #print("Running on image with bbox")
        
        if self.config['ocr']['method'] == 'pytesseract':
            txts = self.pytesseract_image_bbox(image, bboxes)
            return dict(map(lambda i,j : (i,j) , classes,txts))
        elif self.config['ocr']['method'] == 'paddleocr':
            self.paddleocr_recog = model
            labelling = self.paddleocr_ssd_compare(image, bboxes,classes)
            return labelling
        elif self.config['ocr']['method'] == 'trocr':
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
            self.trocr_model = (VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')).cpu()
            txts = self.trocr_image_bbox(image, bboxes)
            return dict(map(lambda i,j : (i,j) , classes,txts))
        elif self.config['ocr']['method'] == 'kerasocr':
            self.keras_ocr_model = model
            start = time.time()
            txts = self.kerasocr_image_bbox(image, bboxes, onnx=False)
            print("OCR inside func: ", time.time()-start)
            return dict(map(lambda i,j : (i,j) , classes,txts))
        elif self.config['ocr']['method'] == 'kerasocr_onnx':
            providers = ['CPUExecutionProvider']
            self.keras_ocr_model = model
            self.kerasocr_onnx = rt.InferenceSession(self.config["ocr"]["kerasocr_onnx_path"], providers=providers)
            start = time.time()
            txts = self.kerasocr_image_bbox(image, bboxes, onnx=True)
            #print("OCR inside func: ", time.time()-start)
            return dict(map(lambda i,j : (i,j) , classes,txts))
        else:
            raise Exception('ocr method not supported')

    def run_image(self, image):
        print("Running on image")
        if self.config['ocr']['method'] == 'pytesseract':
            return self.pytesseract_image(image)
        elif self.config['ocr']['method'] == 'paddleocr':
            return self.paddleocr_image(image)
        elif self.config['ocr']['method'] == 'trocr':
            return self.trocr_image(image)
        elif self.config['ocr']['method'] == 'kerasocr':
            txts = self.kerasocr_image(image)
        else:
            raise Exception('ocr method not supported')

    def preprocess_bbox(self, old_bboxes, h, w):
        bboxes = []

        for box in old_bboxes:
            try:
                box = ast.literal_eval(box)
                xc, yc, wb, hb = box
                xc = xc * w
                yc = yc * h
                wb = wb * w
                hb = hb * h
                bboxes.append([int(xc-wb/2), int(yc+hb/2),
                              int(xc+wb/2), int(yc-hb/2)])
            except Exception as e:
                bboxes.append("No BBox")
        return bboxes

    def draw_bbox(self, img, boxes, labels, scores, method='cv2'):
        new_img = img.copy()
        if method == 'cv2':
            for box, label, score in zip(boxes, labels, scores):
                try:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # + ':' + str(score)
                    cv2.putText(new_img, label+' ('+str(round(score,2))+')', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
                    # cv2.putText(new_img, str(round(score,2)), (x2 - 20, y1-10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 4)
                except:
                    continue
        else:
            raise Exception('method not supported')
        return new_img

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        k = round(250/gray.shape[0])
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        ret,gray=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        angle, rotated_gray = self.correct_skew(gray)
        gray = cv2.erode(rotated_gray,(5,5),iterations=10)
        gray= cv2.medianBlur(gray,5)
        gray = cv2.copyMakeBorder(gray, gray.shape[0]//4, gray.shape[0]//4, gray.shape[1]//6, gray.shape[1]//6, cv2.BORDER_CONSTANT)
        return gray

    def correct_skew(self, image, delta=1, limit=5):
        def determine_score(arr, angle):
            data = ndimage.rotate(arr, angle)
            histogram = np.sum(data, axis=1)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
            return histogram, score
        scores = []
        angles = np.arange(-limit, limit + delta, delta/2)
        for angle in angles:
            histogram, score = determine_score(image, angle)
            scores.append(score)
        best_angle = angles[scores.index(max(scores))]
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE)
        return best_angle, rotated


    def pytesseract_image(self, image):
        text = pytesseract.image_to_string(image,lang='eng', config=' --psm 8 --oem 3 -c tessedit_char_whitelist=0123456789')
        return text

    def pytesseract_image_bbox(self, img, bboxes):
        labels = []

        for i, box in enumerate(bboxes):
            if box != "No BBox":
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2-x1, y2-y1
                crop = img[y:y+h, x:x+w]

                text = self.pytesseract_image(crop)
                labels.append(text)
            else:
                labels.append("NA")
        return labels

    def kerasocr_image(self, image):
        text = self.keras_ocr_model.recognize(image)
        return text
    
    def kerasocr_onnx_image(self, image):
        h, w, c = self.keras_ocr_model.model.input_shape[1], self.keras_ocr_model.model.input_shape[2], self.keras_ocr_model.model.input_shape[3]
        image = keras_ocr.tools.read_and_fit(image, width=w, height=h)
        image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        image = image.astype("float32") / 255
        image = image[np.newaxis]

        onnx_pred = self.kerasocr_onnx.run(['decode'], {"input": image})[0][0] #output_names
        #onnx_pred = [np.argmax(i) for i in onnx_pred]
        text = "".join(
                    [
                        self.keras_ocr_model.alphabet[idx]
                        for idx in onnx_pred
                        if idx not in [len(self.keras_ocr_model.alphabet), -1]
                    ]
                )

        return text

    def kerasocr_image_bbox(self, image, bbox, onnx):
        labels = []
        for i, box in enumerate(bbox):
            if box != "No BBox":
                try:
                    x1, y1, x2, y2 = box
                    x, y, w, h = x1, y1, x2-x1, y2-y1
                    crop = image[y:y+h, x:x+w]
                    if onnx:
                        text = self.kerasocr_onnx_image(crop)
                    else:
                        text = self.kerasocr_image(crop)
                    labels.append(text)
                except Exception as e:
                    print(e)
                    labels.append("NA")
            else:
                labels.append("NA")
        return labels

    def paddleocr_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        cv2.imwrite("./tmp.png", cv2.cvtColor(dilation, cv2.COLOR_RGB2BGR))
        result = self.paddleocr_model.ocr("./tmp.png")[0]
        text = " ".join([line[1][0] for line in result])
        os.remove("./tmp.png")
        return text

    def paddleocr_image_bbox(self, img, bboxes):
        labels = []

        for box in bboxes:
            if box != "No BBox":
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2-x1, y2-y1
                crop = img[y:y+h, x:x+w]

                text = self.paddleocr_image(crop)
                labels.append(text)
                os.remove("./tmp.png")
            else:
                labels.append("NA")
        return labels

    def paddleocr_model(self,image):
      result = self.paddleocr_recog.ocr(image)
      return result
    
    def filter_top_bbox(self, bboxes, txts, topk=20):
        areas = []
        for bbox in bboxes:
            x1, y1, x2, y2 = int(bbox[3][0]), int(bbox[3][1]), int(bbox[1][0]), int(bbox[1][1])
            x, y, w, h = x1, y2, x2-x1, y1-y2
            areas.append(w*h)
        
        top_ids = np.argsort(areas)[::-1][:topk]
        top_bboxes, top_txts = [], []
        
        for i in top_ids:
            bbox = bboxes[i]
            x1, y1, x2, y2 = int(bbox[3][0]), int(bbox[3][1]), int(bbox[1][0]), int(bbox[1][1])
            x, y, w, h = x1, y2, x2-x1, y1-y2
            if h > 40:
                top_bboxes.append(bboxes[i])
                top_txts.append(txts[i])
        
        return top_bboxes, top_txts

    def paddleocr_ssd_compare(self,image,ssd_boxes,ssd_txts):
      result=self.paddleocr_model(image)
      boxes =[line[0] for line in result[0]]
      txts = [line[1][0] for line in result[0]] #[1][0]
      scores = [line[1][1] for line in result[0]] #[1][1]
      top_boxes,top_txts=self.filter_top_bbox(boxes,txts)
      updated_ocr_output=[]
      updated_ssd_output=[]
      for box,txt in zip(top_boxes,top_txts):
         updated_ocr_output.append({'bbox':box, 'label_name': txt})
      for box,txt in zip(ssd_boxes,ssd_txts):
         updated_ssd_output.append({'bbox':box, 'label_name': txt})
      final_label=self.paddleocr_ssd_distance(updated_ocr_output,updated_ssd_output)
      return final_label
    
    def paddleocr_ssd_distance(self,ocr_output,ssd_output):
      labelling={}
      for j in ssd_output:
          box=j['bbox']
          distances=[]
          x1, y1, x2, y2 = j['bbox']
          Xcenter_ssd,Ycenter_ssd=(x2+x1)/2,(y2+y1)/2
          for k in ocr_output:
             X1,Y1,X2,Y2=int(k['bbox'][0][0]), int(k['bbox'][0][1]), int(k['bbox'][2][0]), int(k['bbox'][2][1])
             Xcenter_ocr,Ycenter_ocr=(X2+X1)/2,(Y2+Y1)/2
             dist=math.sqrt( ((Xcenter_ocr-Xcenter_ssd)**2)+((Ycenter_ocr-Ycenter_ssd)**2) )
             distances.append(dist)
          min_dist=min(distances)
          minposition = distances.index(min(distances))
          labelling[j['label_name']] = ocr_output[minposition]['label_name']
      return labelling

    def ssd_bbox_filter(self,boxes,txts):
      updated_ssd_output=[]
      for box,text in zip(boxes,txts):
        if type(text) == str:
           updated_ssd_output.append({'bbox':box, 'label_name': text})
      return updated_ssd_output

    
    def paddleocr_bbox_filter(self,boxes,txts):
      updated_ocr_output=[]
      for box,txt in zip(boxes,txts):
        if txt.isnumeric():
          updated_ocr_output.append({'bbox':box, 'label_name': txt})
        else:
          pass
      return updated_ocr_output

    def trocr_image(self, image):
        pixel_values = self.trocr_processor(
            images=image, return_tensors="pt").pixel_values
        generated_ids = self.trocr_model.generate(pixel_values)
        text = self.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True)
        return text

    def trocr_image_bbox(self, img, bboxes):
        labels = []
        ls = []
        img_array = []

        for i, box in enumerate(bboxes):
            if box != "No BBox":
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y1, x2-x1, y2-y1
                crop = img[y:y+h, x:x+w]
                pixel_values = self.trocr_processor(
                    images=crop, return_tensors="pt").pixel_values
                img_array.append(pixel_values)
                ls.append(i)
            else:
                continue

        img_array = torch.cat(img_array, axis=0).cpu()

        generated_ids = self.trocr_model.generate(img_array)
        text = self.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True)

        count = 0
        for j in range(len(bboxes)):
            if j in ls:
                labels.append(text[count])
                count = count + 1
            else:
                labels.append("NA")
        return labels