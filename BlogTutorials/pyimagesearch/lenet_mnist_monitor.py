import matplotlib

from BlogTutorials.pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from BlogTutorials.pyimagesearch.nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
import numpy as np
import argparse
import os
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to the output directory")
args = vars(ap.parse_args())
print("[info process ID: {}".format(os.getpid()))


# load MNIST dataset ~ 55 MB download
print("[info] loading MNIST dataset...")
dataset = datasets.fetch_openml('mnist_784', version=1, cache=True)
data = dataset.data
print("[info] laoded MNIST dataset complete")

# reshape data from 1d (784,) array to 28x28
if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)

# Scale and train/test splits
(trainX, testX, trainY, testY) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)

le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

print(" compile model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the set of callbacks during training
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print(" training...")
epochs = 20
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=epochs, callbacks=callbacks, verbose=1)

print("evaluation ...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))



