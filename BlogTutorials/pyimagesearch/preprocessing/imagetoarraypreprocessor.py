from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data fromat, channels first or last of image
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges
        # the dimensions of the image, converts image to numpy array of
        # (row, columns, color_changel)
        return img_to_array(image, data_format=self.dataFormat)
