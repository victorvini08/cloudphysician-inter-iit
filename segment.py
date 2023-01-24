import cv2
import numpy as np
import torch
from torchvision import transforms as tt
import segmentation_models_pytorch as smp
from imutils import perspective

class Segment:
    def __init__(self, config):
        self.config = config
        self.cuda = torch.cuda.is_available() 
        if not self.cuda:
            print('CUDA Not Available, running on CPU.\n')
        model = torch.load(config['unet-model-path'], 'cpu')
        self.unet = smp.Unet(encoder_weights = model['encoder_weights'],
                            encoder_depth = model['encoder_depth'],
                            encoder_name = model['encoder_name'],
                            decoder_channels = model['decoder_channels'],
                            classes = 2)
        if self.cuda:
            self.unet = self.unet.cuda()

        self.unet.load_state_dict(model['state_dict'])
        self.transforms = tt.Compose([
            tt.Resize((192, 320)),
            tt.ToTensor()
        ])
    
    def predict_mask(self, image, w, h):
        pred = self.unet(image)[0].detach().cpu().numpy()
        pred = np.argmax(pred, axis=0)

        pred = np.reshape(pred, (pred.shape[0], pred.shape[1], 1)).astype(np.float32)
        pred = cv2.resize(pred, (w, h))

        return pred


    def find_box(self, pred):
        contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_area_idx = 0
        for i,contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_area_idx = i

        try:
            epsilon = 0.1*cv2.arcLength(contours[max_area_idx],True)
            approx = cv2.approxPolyDP(contours[max_area_idx],epsilon,True)
            box = np.reshape(approx, (4, 2))
        except:
            box = cv2.minAreaRect(contours[max_area_idx])
            box = cv2.boxPoints(box)
            box = np.array(box, dtype='int32')
        
        box = perspective.order_points(box).astype(np.int32)
        box[0] -= 10
        box[1][0] += 10
        box[1][1] -= 10
        box[2] += 10
        box[3][0] -= 10
        box[3][1] += 10

        return box


    def crop_image(self, pred, original, box):

        mask = np.zeros(pred.shape)
        mask = cv2.fillPoly(mask, pts = [box], color = (1,1,1))

        original[:,:,0] = original[:,:,0]*mask
        original[:,:,1] = original[:,:,1]*mask
        original[:,:,2] = original[:,:,2]*mask
        return original


    def run(self, image):
        """
        Arguments
            image: Numpy Array of Image in RGB: (H, W, 3)

        Returns
            ROI: Numpy Array of Region of Interest in RGB: (H, W, 3)
        """
        h,w,c = image.shape
        assert len(image.shape) == 3, "Segmentation Module Error - Image must be a 3 dimensional array"
        assert image.shape[2] == 3, "Segmentation Module Error - Image must have 3 color channels"

        original = image
        image = self.transforms(Image.fromarray(image))
        image = image.unsqueeze(0)
        if self.cuda:
            image = image.cuda()

        pred = self.predict_mask(image, w, h)
        box = self.find_box(pred)
        image = self.crop_image(pred, original, box)

        target = np.array([[0,0], [w-1,0], [w-1,h-1], [0, h-1]])
        matrix = cv2.getPerspectiveTransform(box.astype(np.float32), target.astype(np.float32))
        image = cv2.warpPerspective(image, matrix, (w,h))

        return image