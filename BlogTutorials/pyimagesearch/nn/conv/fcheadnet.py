from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        """
        Complete a transfer learning with a FC head layer. as described:
        INPUT => fc => RELU => DO => FC => SOFTMAX
        :param baseModel: model without FC head (VGG, resnet, etc.)
        :param classes: number of output classes to classify
        :param D: Number of hidden nodes in the FC layer after input
        :return: headModel (baseModel with new set of FC layers on end
        """
        # init base model and add our head of FC layers
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel



