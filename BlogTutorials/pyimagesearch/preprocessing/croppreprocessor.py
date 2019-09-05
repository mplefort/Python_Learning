# crops a 4 corner and center and corresponder horizontal flip
import numpy as np
import cv2

class CropPreprocessor:
    def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
        '''
         4 corner and center crop an image with horizontal flip (form of data augmentation)
         for 1-5% normal classification increase
        :param width: crop width
        :param height: crop height
        :param horiz: if horizontal flips should be included
        :param inter: type of cv2 interpolation
        '''
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter

    def preprocess(self, image):
        '''
            preprocess the image and create 4 corner and center crop an image with horizontal flip (form of data augmentation)
            for 1-5% normal classification increase

        :param image: iamge to process
        :return: images:
        '''
        crops = []

        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h] ]

        # compute the center crop
        dW = int(0.5 * (w - self.width))
        dH = int(0.5 * (h - self.height))
        coords.append([dW, dH, w - dW, h - dH])

        # extract crops from coords
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.wdith, self.height), interpolation=self.inter)
            crops.append(crop)

        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)

