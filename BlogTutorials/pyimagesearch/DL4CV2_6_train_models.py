# Train an ensemble (multiple) networks for improved accuracy using a forloop
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer #(convert labels to array of 0/1import
from sklearn.metrics import classification_report
from BlogTutorials.pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10

import matplotlib.pyplot as plt
import numpy as np
import os

"""
Train multiple versions of MiniVGGnet ensemble to improve accuracy.
"""
# Paths
output = r'D:\matth\Documents\projects\python\models\miniVGG_ensemble_cifar10'
model_path = r'D:\matth\Documents\projects\python\models\miniVGG_ensemble_cifar10'
num_models = 5 # number models to train (random forests use 30-100, CNNs 5-10 average)

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# add data augmentation with image geneartro from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=10, # randomly rotate +/- 30 deg
                         width_shift_range=0.1, # translate by 0.1
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Loop to train number of models
for i in np.arange(0, num_models):
    print("[info] training model {} / {}".format(i+1, num_models))
    opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(32,32,3,10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train model and save
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                            validation_data=(testX, testY),
                            epochs=40,
                            steps_per_epoch=len(trainX)//64,
                            verbose=1)

    p = [model_path, "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    # evalueate and save report of each network for review
    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=labelNames)

    p = [model_path, "model_{}.txt".format(i)]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()

    # plot each report

    p = [model_path, "model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()


