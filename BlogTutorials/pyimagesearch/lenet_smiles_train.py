from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from BlogTutorials.pyimagesearch.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

dataset_path = 'D:\matth\Documents\projects\python\datasets\SMILEsmileD\SMILEs'
output_path = 'D:\matth\Documents\projects\python\models\lenet_smiles_model\lenet_smiles.hdf5'

data = []
labels = []

for imagePath in sorted(list(paths.list_images(dataset_path))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

# scale the raw pixel values to 0-1
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from ints to vectors with one hot encoding
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# accouunt for the skew in data labels. Used to amplifiy training weights of "smiling" cae given the ratio of
# [9475 to 3690] (non smiling to smiling)
classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals

# split data (stratify sampling samples at same ratio of data set for test to train split. i.e the 9475:3690 ratio
#  is kept in both the test and training set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.2, stratify=labels, random_state=42)

# Train the network - adam optimizer for faster convergence, class_weights used for data sample imbalance
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("[info] training network...")
epochs = 15
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=classWeight,
              batch_size=64, epochs=epochs, verbose=1)

# predict and evaluation
print("[info] evaluation....")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

print("[info] saving mdoel...")
model.save(output_path)

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






