from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import add
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        """

        see DL4CV [46] Kaiming He. Deep Residual Networks github.com/KaimingHe/deep-residual-networks
        :param data: input to residual model (prev layer)
        :param K: Filters in the convultions 1x1x(K/4) -> 3x3x(K/4) -> 1x1xK
        :param stride: stride in convolutions
        :param chanDim: Channel Dimension (first or last)
        :param red: "reduce" bool control whether module will reduce spatial dims.
        :param reg: regularization to apply to CONVs
        :param bnEps: epsilon to avoid divide by zero (Keras defaults to 1e-3)
        :param bnMom: momentum of batch normalization moving average. (Keras defaults to 0.99)
        :return: data to next module
        """

        # shortcut right branch to add at end
        shortcut = data

        # Conv block 1/3 (1x1xK/4 filters)
        bn1 = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(data)
        act1 = Activation("relu")(bn1)
        # bias left out as He et al. bias in BN layer following conv. no need to include twice
        conv1 = Conv2D(filters=int(K * 0.25), kernel_size=(1,1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)

        # Conv block 2/3 (3x3xK/4 filter)
        bn2 = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(bn1)
        act2 = Activation("relu")(bn2)
        # bias left out as He et al. bias in BN layer following conv. no need to include twice
        conv2 = Conv2D(filters=int(K * 0.25), kernel_size=(3,3), strides=stride, padding="same",
                       use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # Conv block 3/3 (1x1xK filters)
        bn3 = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(conv2)
        act3 = Activation("relu")(bn3)
        # bias left out as He et al. bias in BN layer following conv. no need to include twice
        conv3 = Conv2D(filters=(K), kernel_size=(1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act3)

        # if reduce dims apply conv to shortcut with stride > 1
        if red:
            shortcut = Conv2D(filters=K, kernel_size=(1,1), strides=stride,
                              use_bias=False, kernel_regularizer=l2(reg))(act1)

        x = add([conv3, shortcut])
        return x


    @staticmethod
    def build(width, height, depth, classes, stages,
              filters, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
        """

        :param width: (int) input image width
        :param height: (int) input image height
        :param depth: (int) input image depth
        :param classes: (int) number classes
        :param stages: [(int)] each stage num of repeated residual modules use. len(stages) = n
        :param filters: [(int)] num of filters to use for each module, n=1 filter applied to
                        init conv layer. [1:] filters apply to stages. len(filters) = n+1
        :param reg: (float) regularization for Conv
        :param bnEps: (float)batch nomalization Epsilon (avoid dividing by zero)
        :param bnMom: (float) batch normaliztion Mom (rolling average)
        :param dataset: (str)
        :return: keras.model.Model
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
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        #if cifar dataset apply a single conv
        if dataset == "cifar":
            x = Conv2D(filters[0], (3,3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i+1], stride, chanDim,
                                       red=True, bnEps=bnEps, bnMom=bnMom)

            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i + 1], (1,1), chanDim,
                                           red=True, bnEps=bnEps, bnMom=bnMom)

        # will reduce to (8x8xclasses)???
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="resnet")

        return model
