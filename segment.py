import cv2

class segment:
    def __init__(self,config):
        self.config = config 
        self.model = None
    
    def run(self,image):
        img = cv2.imread(image)
        
        return img
        #return the segmented image
