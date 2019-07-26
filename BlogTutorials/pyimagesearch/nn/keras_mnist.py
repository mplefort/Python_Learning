from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np
import argparse
# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load MNIST dataset ~ 55 MB download
print("[info] loading MNIST dataset...")
dataset = datasets.fetch_openml('mnist_784', version=1, cache=True)
print("[info] laoded MNIST dataset complete")

# scale data to [0  1.0[
data = dataset.data.astype("float") / 255.0
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Define NN model from Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[info] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

print("[info] evaluating network")
predictions = model.predict(testX, batch_size=256)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,100), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Acc")
plt.legend()
plt.savefig(args["output"])