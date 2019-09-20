from keras.layers.normalization import BatchNormalization # applied before or after Activation function
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate # concatenate layer outputs along the channel dim
from keras.models import Model  # ilo of sequential where output of 1 is input to next
from keras import backend as K
from keras.regularizers import l2  #consider adding to see reduction of overfitting benefit, see mini2_googlenet for implementation


class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
        """
            Note: Model class ilo of sequential remove model.add() method.
            instead "Functional API" from keras is used with
                    output = Layer(params)(input)

            :param x: input layer
            :param K: number for filters for convolution
            :param kX: size of each filter
            :param kY: size of each filter
            :param stride: stride of CONV layer
            :param chanDim: channels last or first ordering
            :param padding: type of padding
            :return: output of block
        """
        # defines a CONV -> BN -> RELU pattern (consider swapping BN and RELU)
        x = Conv2D(filters=K, kernel_size=(kX, kY), strides=stride, padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):
        # define two CONV modules, then concatenat across the channel Dim
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1,1), chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1,1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
        return x

    @staticmethod
    def downsample_module(x, K, chanDim):
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
        max_pool = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid")(x)
        x = concatenate([conv_3x3, max_pool], axis=chanDim)
        return x

    @staticmethod
    def build(width, height, depth, classes):
        # init the input shape to be "channels last" and the channels dims iteslf,
        inputShape = (height, width, depth)
        chanDim = -1

        # if using channels first correct it now
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        #define mdoel input and frist conv module
        inputs = Input(shape=inputShape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim, padding="same")

        # two inceptsion and downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim)

        # 4 inceptions and a downsample
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

        # 2 inceptions, 1 mean pooling, FC
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # softmax classifer, not output of inception_module (176, 160) is (7,7,336)
        # so only 1 FC layer of dense from final output of (1,1,336) after averagepooling
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create model
        model = Model(inputs, x, name="minigooglenet")

        return model
