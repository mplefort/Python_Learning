from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

from keras.regularizers import l2 # L2 normalization weight decay to cost function
from keras import backend as K

class ShallowNet2:

    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
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
        model.add(Conv2D(filters=32, kernel_size=(3,3),  input_shape=inputShape, kernel_regularizer=l2(reg))) # padding="same",
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2))) #,
                               #padding='valid'))
        # model.add(Dropout(0.25))

        # 32 K filters of 3,3 size, padding for same output size,
        model.add(Conv2D(filters=64, kernel_size=(3,3),  kernel_regularizer=l2(reg))) # padding="same",
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2))) #,
                               #padding='valid'))
        # model.add(Dropout(0.25))

        # 32 K filters of 3,3 size, padding for same output size,
        model.add(Conv2D(filters=128, kernel_size=(3,3),  kernel_regularizer=l2(reg))) # padding="same",
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2))) #,
                               #padding='valid'))
        # model.add(Dropout(0.25))

        # 32 K filters of 3,3 size, padding for same output size,
        model.add(Conv2D(filters=128, kernel_size=(3,3),  kernel_regularizer=l2(reg))) # padding="same",
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2))) #,
                               #padding='valid'))
        # model.add(Dropout(0.25))

        # 4. FC => RELU layer
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(units=1024,
                        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        # model.add(BatchNormalization())

        # 5. FC => RELU
        # model.add(Dense(units=1024,
        #                 kernel_regularizer=l2(reg)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        # 6. softmax classifer
        # if classes == 2:
        #     model.add(Dense(1, activation='sigmoid'))
        # else:
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))


        return model


