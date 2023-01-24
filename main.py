import yaml
from ocr import ocr_model
from segment import Segment
from detect import detect
import cv2

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


img = '/home/raj/cloudphy/main-pipeline/sample_images/aastha_icu_mon--5_2023_1_2_9_0_0.jpeg'
image = cv2.imread(img)

seg = Segment(config)
ocr = ocr_model(config)
detection = detect(config)

image = seg.run(image)
ans = ocr_model.run_image(image)
print(ans)


        