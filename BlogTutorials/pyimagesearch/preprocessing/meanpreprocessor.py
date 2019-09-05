# subtract RGB mean from an image
import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean  = rMean
        self.gMean  = gMean
        self.bMean  = bMean

    def preprocess(self, image):
        '''
         Subtracts mean rgb from each pixel and returns image
        :param image: image to subtract RGB means from
        :return: image
        '''
        (B, G, R) = cv2.split(image.astype("float32"))
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean

        return cv2.merge([B, G, R])