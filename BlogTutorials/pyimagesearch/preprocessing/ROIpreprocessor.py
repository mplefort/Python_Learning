import numpy as np
import cv2


class ROIPreprocessor:
    def __init__(self, xmin, xmax, ymin, ymax, inter=cv2.INTER_AREA):
        # crop image to window defined as [minx:maxx, miny:maxy] interoploation method
        self.minx = xmin
        self.miny = ymin
        self.maxx = xmax
        self.maxy = ymax

    def preprocess(self, image):

        if self.minx < 0 or self.miny < 0 or self.maxx > image.shape[0] or self.maxy > image.shape[1]:
            raise("ROI outside of range of image range")

        return image[self.minx:self.maxx, self.miny:self.maxy]
