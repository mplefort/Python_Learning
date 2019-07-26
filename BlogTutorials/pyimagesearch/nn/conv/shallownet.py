from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class ShallowNet:

    @staticmethod
    def build(width, height, depth, classes):
        """
            Build shallow net arch: INPUT => CONV => RELU => FC
            :param
                width: width  of input images (cols in matrix) \n
                height: height of input images (rows in matrix) \n
                depth: num of channels in input image \n
                classes: num of classes network should learn to predict

            :return
                model
        """
        # initialize the model along with the input shape to be
        # "channels first"
        model = Sequential()
        # inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        else:
            inputShape = (height, width, depth)

        # 32 K filters of 3,3 size, padding for same output size,
        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


