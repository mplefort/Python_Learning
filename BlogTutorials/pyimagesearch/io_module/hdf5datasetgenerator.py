#Open a dataset in batches for training when dataset too large to fit into memeory at one time

from keras.utils import np_utils
import numpy as np
import h5py

class HD5FDatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessor=None,
                 aug=None, binarize=True, classes=2):
        '''

        :param dbPath: path to HDF5 dataset with images and corresponding labels
        :param batchSize: size of batches to train on
        :param preprocessor: list of image preprocessores (MeanPreprocessor, ImageToArrayPreprocessor, etc.))
        :param aug: Default = None, or use Keras ImageDataGenerator to apply data augmentation
        :param binarize: one hot encoding for "categorical-crossentropy" or "binary cross-entropy"
        :param classes: Default=2, used to construct binarize
        '''
        self.batchSize = batchSize
        self.preprocessors = preprocessor
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # open the HDF% databaase for reading and get number of entries
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        '''
            yields batches of images for Keras .fit_generator function
        :param passes: number of epochs to train for
        :return:
        '''

        epochs = 0

        # loop infinitely until number of epochs reached
        while epochs < passes:
            print("[info] data generator on epoch {} / {}".format(epochs+1, passes))
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract images and lable of batchsize
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                if self.binarize:
                    labels = np_utils.to_categorical(y=labels,
                                                     num_classes=self.classes)

                if self.preprocessors is not None:
                    procImages = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    # overwrite image to be a processed image
                    images = np.array(procImages)

                # check if data aug needed
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))
                yield (images, labels)

            epochs += 1

    def close(self):
        self.db.close()