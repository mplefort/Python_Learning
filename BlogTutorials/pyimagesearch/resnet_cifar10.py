import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from BlogTutorials.pyimagesearch.nn.conv.resnet import ResNet
from BlogTutorials.pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from BlogTutorials.pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.models import load_model

import keras.backend as K
import numpy as np
import sys


checkpoint_path = r"D:\matth\Documents\projects\python\models\resnet_cifar10\checkpoints"
plot_path = r"D:\matth\Documents\projects\python\models\resnet_cifar10\resnet56_cifar10.png"
json_path = r"D:\matth\Documents\projects\python\models\resnet_cifar10\resnet56_cifar10.json"
model_load_path = None
# model_load_path = r"D:\matth\Documents\projects\python\models\resnet_cifar10\checkpoints"
starting_epoch = 0
init_lr  = 1e-1
num_epochs = 100
batch_size = 64
def poly_decay(epoch):
    maxEpochs = num_epochs
    baseLr = init_lr
    power = 1.0
    alpha = baseLr * (1 - (epoch / float(maxEpochs))) ** power
    return alpha



# load CIFAR-10 Data
print("Loading CIFAR 10")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtration
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# apply data augmentation at input during training
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

if model_load_path is None:
    print("[info] compiling model...")
    opt = SGD(lr=init_lr)
    model = ResNet.build(32, 32, 3, classes=10, stages= (12, 8, 4),
                          filters=(64, 64, 128, 256), reg=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

else:
    print("loading model {} ...".format(model_load_path))
    model = load_model(model_load_path)

    print("[info] old learning rate {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, init_lr)
    print("[info] new learning rate {}".format(K.get_value(model.optimizer.lr)))

callbacks = [EpochCheckpoint(checkpoint_path, every=5, startAt=starting_epoch),
             TrainingMonitor(plot_path, jsonPath=json_path, startAt=starting_epoch)]

# train network
model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(testX, testY) )
