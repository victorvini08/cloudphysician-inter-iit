import os
import cv2
import torch
import  numpy as np

from paddleocr import PaddleOCR
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class ip:
    def __init__(self) -> None:
        self.model = None
               
    
    def run(image,lower,upper,class_type = 'hr'):
        mask=cv2.inRange(image,low_limit_spo2,upper_limit_spo2)
        res=cv2.bitwise_and(image,image,mask=mask)
        return image
        
    def threshold_values(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur=cv2.medianBlur(gray,9)
        _,thresh=cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
        cnts,hier=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                
        mask = np.zeros(img.shape[0:2], np.uint8)
        cv2.drawContours(mask, cnts, -1, 255, -1)
        b,g,r=cv2.split(img)

        image=cv2.bitwise_and(img,img,mask=thresh)
        b_mean = cv2.mean(b, mask=mask)
        g_mean = cv2.mean(g, mask=mask)
        r_mean = cv2.mean(r, mask=mask)
        
        avg_value=[b,g,r]
        return avg_value
    

class ocr_model:
    def __init__(self, config):
        self.config = config

        self.paddleocr_model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

    def run_image(self, image):
        print("Running on image")
        images = self.image_masking(image)
        if self.config['ocr']['method'] == 'pytesseract':
            return self.pytesseract_image(images)
        elif self.config['ocr']['method'] == 'paddleocr':
            return self.paddleocr_image(images)
        elif self.config['ocr']['method'] == 'trocr':
            return self.trocr_image(images)
        else:
            raise Exception('ocr method not supported')


    def pytesseract_image(self, images):
        labels = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
            labels.append(pytesseract.image_to_string(dilation))
        return labels


    def paddleocr_image(self, images):
        labels = []
        for img in images:
            cv2.imwrite("./tmp.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            result = self.paddleocr_model.ocr("./tmp.png", cls=True)[0]
            labels.append(" ".join([line[1][0] for line in result]))
            os.remove("./tmp.png")
        return labels

    def trocr_image(self, images):
        labels, img_array = [], []
        for path in images:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pixel_values = self.trocr_processor(
                    images=img, return_tensors="pt").pixel_values
            img_array.append(pixel_values)


        img_array = torch.cat(img_array, axis=0)

        generated_ids = self.trocr_model.generate(img_array)
        labels = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)

        return labels

    def image_masking(self, image):
        # for HR values
        low_limit_hr = np.array([0, 192, 0])
        upper_limit_hr = np.array([199, 255, 172])
        mask_hr = cv2.inRange(image, low_limit_hr, upper_limit_hr)
        hr = cv2.bitwise_and(image, image, mask=mask_hr)
        # for RR values
        low_limit_rr = np.array([56, 208, 209])
        upper_limit_rr = np.array([204, 255, 255])
        mask_rr = cv2.inRange(image, low_limit_rr, upper_limit_rr)
        rr = cv2.bitwise_and(image, image, mask=mask_rr)
        # for SpO2 values
        low_limit_spo2 = np.array([216, 208, 57])
        upper_limit_spo2 = np.array([255, 255, 210])
        mask_spo2 = cv2.inRange(image, low_limit_spo2, upper_limit_spo2)
        spo2 = cv2.bitwise_and(image, image, mask=mask_spo2)
        # for BP values
        low_limit_bp = np.array([214, 204, 217])
        upper_limit_bp = np.array([255, 255, 255])
        mask_bp = cv2.inRange(image, low_limit_bp, upper_limit_bp)
        bp = cv2.bitwise_and(image, image, mask=mask_bp)
        images = [hr, bp, rr, spo2]

        return images

    
