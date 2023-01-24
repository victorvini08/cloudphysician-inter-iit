import os
import ast
import cv2
import torch

#from paddleocr import PaddleOCR
#import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class ocr_model:
    def __init__(self, config):
        self.config = config
        self.paddleocr_model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

    def run_bbox(self, image, bboxes):
        h, w, _ = image.shape
        bboxes = self.preprocess_bbox(bboxes, h, w)
        print("Running on image with bbox")

        if self.config['ocr']['method'] == 'pytesseract':
            return self.pytesseract_image_bbox(image, bboxes)
        elif self.config['ocr']['method'] == 'paddleocr':
            return self.paddleocr_image_bbox(image, bboxes)
        elif self.config['ocr']['method'] == 'trocr':
            return self.trocr_image_bbox(image, bboxes)
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

    def draw_bbox(self, img, boxes, labels, method='cv2'):
        new_img = img.copy()
        if method == 'cv2':
            for box, label in zip(boxes, labels):
                try:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(new_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # + ':' + str(score)
                    cv2.putText(new_img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                except:
                    continue
        else:
            raise Exception('method not supported')
        return new_img

    def pytesseract_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(
            gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        text = pytesseract.image_to_string(dilation)
        return text

    def pytesseract_image_bbox(self, img, bboxes):
        labels = []

        for i, box in enumerate(bboxes):
            if box != "No BBox":
                x1, y1, x2, y2 = box
                x, y, w, h = x1, y2, x2-x1, y1-y2
                crop = img[y:y+h, x:x+w]

                text = self.pytesseract_image(crop)
                labels.append(text)
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
                x, y, w, h = x1, y2, x2-x1, y1-y2
                crop = img[y:y+h, x:x+w]

                text = self.paddleocr_image(crop)
                labels.append(text)
                os.remove("./tmp.png")
            else:
                labels.append("NA")
        return labels

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
                x, y, w, h = x1, y2, x2-x1, y1-y2
                crop = img[y:y+h, x:x+w]
                pixel_values = self.trocr_processor(
                    images=crop, return_tensors="pt").pixel_values
                img_array.append(pixel_values)
                ls.append(i)
            else:
                continue

        img_array = torch.cat(img_array, axis=0)

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
