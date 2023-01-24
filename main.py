import yaml
from ocr import ocr_model
from segment import segment
from detect import detect

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



seg = segment(config)
ocr = ocr_model(config)
detection = detect(config)




        