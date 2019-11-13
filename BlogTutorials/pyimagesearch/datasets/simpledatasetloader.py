# All datasets on D drive to avoid C drive fill up:
#  D:\matth\Documents\projects\python\datasets

import numpy as np
import cv2
import os
import matplotlib.image as mpimg

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None, img_load="cv"):
        '''
        Load data and store any preprocessors to be used on loaded image data
        :param preprocessors: preprocessors to be applied to loaded data to be applied sequentially in a pipeline
        :param img_load: [str] use cv or matplotlib to load images. "cv" or "mat"
        '''
        self.preprocessors = preprocessors
        self.img_load = img_load
        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        '''
        :param imagePaths: dir path to image folder to load dataset
        :param verbose: show an update for every i-th image loaded
        :return: tupe of data nad label
        s ad np.arrays
        '''
        # Initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class label assuming that our path has the following format:
            # /path/to/dataset/{class_label}/{image}.jpg

            if self.img_load == "cv":
                image = cv2.imread(imagePath)
            else:
                image = mpimg.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check if need to apply a preprocessor
            if self.preprocessors is not None:
                # apply each preprocessor to the image loaded
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as a "feature vector" by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show update every 'verbose' image
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {} / {}".format(i+1, len(imagePaths)))

        # return a tuple of data and labels
        return (np.array(data), np.array(labels))
