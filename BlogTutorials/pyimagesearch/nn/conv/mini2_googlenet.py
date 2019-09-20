from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.regularizers import l2
from keras import backend as K


# See Figure 11.9 of DL4CV2 for arch followed
class Mini2GoogLeNet:

    @staticmethod
    def conv_module(x, K, kX, kY, stride, chanDim, padding="same",
                    reg=0.0005, name=None):
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
            :param padding: type of padding from keras
            :param reg: l2 regularization amount (see keras documentation)
            :param name: name of conv layer
            :return: output of block
        """
        # init the CONV, Bn, and RELU layer names
        (convName, actName, bnName) = (None, None, None)

        if name is not None:
            convName = name + "_conv"
            actName = name + "_act"
            bnName = name + "_bn"

        # defines a CONV -> BN -> RELU pattern (consider swapping BN and RELU)
        x = Conv2D(filters=K, kernel_size=(kX, kY), strides=stride, padding=padding,
                   kernel_regularizer=l2(reg), name=convName)(x)
        x = Activation("elu", name=actName)(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)

        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3reduce, num3x3, num5x5reduce, num5x5, num1x1Proj, chanDim,
                         stage, reg=0.0005):
        """

        :param x: input from previous layer
        :param num1x1: filters to apply in first branch
        :param num3x3reduce: filters for second branch on 1x1 before 3x3 (num3x3reduce < num3x3)
        :param num3x3: filters for second branch on 3x3
        :param num5x5reduce: filters for third branch on 1x1 before 5x5
        :param num5x5: filters for 5x5
        :param num1x1Proj: MaxPooling followed by 1x1 conv filters
        :param chanDim: chanDim
        :param stage: current stage of model
        :param reg: l2 reg to apply to convultion params
        :return: x
        """
        # define the first branch of Inception (1x1 convolutions)
        first = Mini2GoogLeNet.conv_module(x, K=num1x1, kX=1, kY=1,
                                           stride=(1, 1), chanDim=chanDim,
                                           reg=reg, name=stage + "_first")

        # second branch: 1x1 reduction followed by 3x3 expansion
        second = Mini2GoogLeNet.conv_module(x, K=num3x3reduce, kX=1, kY=1,
                                            stride=(1,1), chanDim=chanDim,
                                            reg=reg, name=stage + "_second1")

        second = Mini2GoogLeNet.conv_module(second, K=num3x3, kX=3, kY=3,
                                            stride=(1, 1), chanDim=chanDim,
                                            reg=reg, name=stage + "_second2")


        # third branch: 1x1 reduction followed by 5x5 expansion
        third = Mini2GoogLeNet.conv_module(x, K=num5x5reduce, kX=1, kY=1,
                                           stride=(1,1), chanDim=chanDim,
                                           reg=reg, name=stage + "_third1")

        third = Mini2GoogLeNet.conv_module(third, K=num5x5, kX=5, kY=5,
                                           stride=(1, 1), chanDim=chanDim,
                                           reg=reg, name=stage + "_third2")

        # fourth branch: pool projection max pooling and 1x1 conv
        fourth = MaxPooling2D(pool_size=(3,3), strides=(1,1),
                              padding="same", name=stage + "_pool")(x)

        fourth = Mini2GoogLeNet.conv_module(fourth, K=num1x1Proj, kX=1, kY=1,
                                            stride=(1,1), chanDim=chanDim,
                                            reg=reg, name=stage + "_fourth")

        x = concatenate(inputs=([first, second, third, fourth]), axis=chanDim, name=stage + "_mixed")

        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.0005):
        """

        :param width:
        :param height:
        :param depth:
        :param classes:
        :param reg:
        :return:
        """

        # init the input shape
        inputShape = (height, width, depth)
        chanDim = -1

        # if channels first switch dims
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # See Figure 11.9 of DL4CV2 for arch followed
        inputs = Input(shape=inputShape)

        x = Mini2GoogLeNet.conv_module(inputs, 64, 5, 5, (1,1), chanDim, reg=reg, name="block1")
        x = MaxPooling2D((3,3), strides=(2,2), padding="same", name="pool1")(x)
        x = Mini2GoogLeNet.conv_module(x, 64, 1, 1, (1,1), chanDim, reg=reg, name="block2")
        x = Mini2GoogLeNet.conv_module(x, 192, 3, 3, (1,1), chanDim, reg=reg, name="block3")
        x = MaxPooling2D((3,3), strides=(2,2), padding="same", name="pool2")(x)

        # apply two inception modules
        x = Mini2GoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim, "3a", reg=reg)
        x = Mini2GoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim, "3b", reg=reg)
        x = MaxPooling2D((3,3), strides=(2,2), padding="same", name="pool3")(x)

        # apply five incpetion modueles in stage4
        x = Mini2GoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chanDim, "4a", reg=reg)
        x = Mini2GoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, chanDim, "4b", reg=reg)
        x = Mini2GoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, chanDim, "4c", reg=reg)
        x = Mini2GoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, chanDim, "4d", reg=reg)
        x = Mini2GoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim, "4e", reg=reg)
        x = MaxPooling2D((3,3), strides=(2,2), padding="same", name="pool4")(x)

        # Apply a POOL layer folled by dropout
        x = AveragePooling2D((4,4), name="pool5")(x)
        x = Dropout(0.4, name="do")(x)

        #soft max classifier
        x = Flatten(name="flatten")(x)
        x = Dense(classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)

        # create model
        model = Model(inputs, x, name="mini2GoogLeNet")

        return model


