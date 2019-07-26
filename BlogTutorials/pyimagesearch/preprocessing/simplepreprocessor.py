# All datasets on D drive to avoid C drive fill up:
#  D:\matth\Documents\projects\python\datasets
import cv2


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        '''
        :param width: target width of image output
        :param height: target height of image output
        :param inter: Interpolation method for resizing (INTER_AREA default)
        '''
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        '''
        :param image: resizes image to fixed size, ignores aspect ratio
        :return: resized image
        '''

        return cv2.resize(src=image, dsize=(self.width, self.height), interpolation=self.inter)
