#alexnet from scratch see blueprint online
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

class Alexnet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # check channels last
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # 1. CONV => RELU => POOL layer set
        model.add(Conv2D(filters=96,
                         kernel_size=(11,11),
                         strides=(4,4),
                         padding="same",
                         input_shape=inputShape,
                         kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3),
                               strides=(2,2),
                               padding='valid'))
        model.add(Dropout(0.25))

        # 2. CONV => RELU => POOL
        model.add(Conv2D(filters=256,
                         kernel_size=(5,5),
                         strides=(1,1),
                         padding='same',
                         kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3),
                               strides=(2,2)))
        model.add(Dropout(0.25))

        # 3. CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384, (3,3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(384, (3,3), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3,3),
                               strides=(2,2)))
        model.add(Dropout(0.25))

        # 4. FC => RELU layer
        model.add(Flatten())
        model.add(Dense(units=4096,
                        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # 5. FC => RELU
        model.add(Dense(units=4096,
                        kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # 6. softmax classifer
        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        return model