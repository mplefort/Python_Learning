import matplotlib
matplotlib.use("Agg") # save plots to disk

from sklearn.preprocessing import LabelBinarizer
from BlogTutorials.pyimagesearch.nn.conv.minigooglenet import MiniGoogLeNet
from BlogTutorials.pyimagesearch.callbacks.trainingmonitor import TrainingMonitor

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.metrics import classification_report

import numpy as np
import os
# TODO: try L2 weight regularization to combat overfitting? see seciton 11.3

# Globals
NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    baseLR = INIT_LR
    maxEpochs = NUM_EPOCHS
    power = 1

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    return alpha

model_output = r"D:\matth\Documents\projects\python\models\miniGoogLeNet_cifar10\miniGoogLeNet_cifar10.hdf5"
log_output= r"D:\matth\Documents\projects\python\models\miniGoogLeNet_cifar10"

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

# callbacks
figPath = os.path.sep.join([log_output, "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([log_output, "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath=figPath,
                             jsonPath=jsonPath,
                             startAt=0),
             LearningRateScheduler(poly_decay)]

# init optimizer and model
print("Compiling Model ")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=opt,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# train
print("Training model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=96),
                    steps_per_epoch=len(trainX) // 96,
                    epochs=NUM_EPOCHS,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=(testX, testY) )

print("save network")
model.save(model_output)

# evaulation
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print("[INFO] evaluating mdoel ...")
predictions = model.predict(testX, batch_size=96)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames)
      )
