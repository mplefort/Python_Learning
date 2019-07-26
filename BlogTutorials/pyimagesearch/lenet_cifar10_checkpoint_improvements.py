from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from BlogTutorials.pyimagesearch.nn.conv.lenet import LeNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
from keras.datasets import cifar10
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to weights dir")
args = vars(ap.parse_args())


# load images
print('loading images')
( (trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from ints to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# Init optimzer and model
epochs = 40
opt = SGD(lr=0.01, decay=0.01/epochs, momentum=0.9, nesterov=True)
model = LeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct teh callback to save only the "best" model to disk
# 03d = 3 digitis
# .4f = float for 4 digits?
fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
fname = os.path.sep.join([args["weights"], "bestmodel.hdf5"])
# monitor optios are val_loss, val_acc, train_loss, train_acc, etc. mode="max/min"
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

# train
print("[INFO] training ...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=epochs, callbacks=callbacks,
              verbose=2)

# evaulation
print("[INFO] evaluating mdoel ...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=labelNames))



plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()