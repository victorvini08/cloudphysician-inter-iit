class ip:
    def __init__(self) -> None:
        self.model = None
               
    
    def run(image,lower,upper,class_type = 'hr'):
        mask=cv.inRange(image,low_limit_spo2,upper_limit_spo2)
        res=cv.bitwise_and(image,image,mask=mask)
        return image
        
    def threshold_values(img):
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        blur=cv.medianBlur(gray,9)
        _,thresh=cv.threshold(blur,150,255,cv.THRESH_BINARY)
        cnts,hier=cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
                
        mask = np.zeros(img.shape[0:2], np.uint8)
        cv.drawContours(mask, cnts, -1, 255, -1)
        b,g,r=cv.split(img)

        image=cv.bitwise_and(img,img,mask=thresh)
        b_mean = cv.mean(b, mask=mask)
        g_mean = cv.mean(g, mask=mask)
        r_mean = cv.mean(r, mask=mask)
        
        avg_value=[b,g,r]
        return avg_value
