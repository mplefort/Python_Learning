from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.aspectawarepreprocessing import AspectAwarePreprocessor
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from BlogTutorials.pyimagesearch.nn.conv.fcheadnet import FCHeadNet

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths

import numpy as np
import os
import matplotlib.pyplot as plt
"""
A complete fine tuning of optimiziing a NN for flowers dataset. Includes:
1. Data Augmentation
2. Transfer learning from VGG16 with FChead
3.
"""
dataset_path = r'D:\matth\Documents\projects\python\datasets\17flowers'
output_modelpath = r'D:\matth\Documents\projects\python\models\vgg16_imagenetTransfer_flowers17_FChead\model.hdf5'

aug = ImageDataGenerator(rotation_range = 30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest")

# get images from disk
print("[info] loading images...")
imagePaths = list(paths.list_images(dataset_path))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

#  init image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Partition training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
# convert the labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Set up network
baseModel = VGG16(weights="imagenet",
                  include_top=False,
                  input_tensor=Input(shape=(224, 224, 3)))
# init new head of network, a set of FC layers follwed by sfotmax FC
headModel = FCHeadNet.build(baseModel,
                            len(classNames),
                            256)

# place head of FC ontop of baseModel
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze layers base layers for training
for layer in baseModel.layers:
    layer.trainable = False

# compile model (after freezing base layer)
print("[info] compile model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# train head of network for a few epochs,
print("[info] training head...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY),
                    epochs=25,
                    steps_per_epoch=len(trainX) // 32,
                    verbose=1)

# Pause after "warm up" of FC head to evaluate
print("[info] evaluatin after init head training...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

# unfreeze latter halft of conv [15 - 19] last conv2D, conv2D, conv2D, MaxPooling2D, Flatten
for layer in baseModel.layers[15:]:
    layer.trainable = True


# recompile model this time with SGD
print("[info] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# train model with last set of basemodel and head unfrozen
print("[info] training head and latter part of base...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY),
                    epochs=100,
                    steps_per_epoch=len(trainX) // 32,
                    verbose=1)

print("[info] evaluating after fine tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

# save model
print("[info] saving model...")
model.save(output_modelpath)

epochs = 100
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
