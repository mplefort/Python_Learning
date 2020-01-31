from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from BlogTutorials.pyimagesearch.preprocessing.aspectawarepreprocessing import AspectAwarePreprocessor

from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from BlogTutorials.pyimagesearch.nn.conv.shallownet import ShallowNet
from BlogTutorials.pyimagesearch.nn.conv.shallownet2 import ShallowNet2
from BlogTutorials.pyimagesearch.nn.conv.alexnet import Alexnet

from BlogTutorials.pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model






from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_path = r"H:\Datasets\animals\data"
OUTPUT_PATH = r"H:\Datasets\animals\models"
epochs = 100

# get class labels
imagePaths = list(paths.list_images(dataset_path))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# image preprocessors
aap = AspectAwarePreprocessor(148, 148)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel to [0,1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths=imagePaths, verbose=500)

im_rgb = data[0].astype("uint8")
plt.imshow(im_rgb)
plt.waitforbuttonpress(timeout=-1)
plt.close('all')


data = data.astype("float") / 255.0


# add data augmentation with image generator from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=40.0, # randomly rotate +/- 30 deg
                         width_shift_range=0.2, # translate by 0.1
                         height_shift_range=0.2,
                         shear_range=0.0, # shear by 0.2
                         zoom_range=0.0, # zoom by [0.8 - 1.2] factory
                         horizontal_flip=True,
                         # brightness_range=[0.9, 1.1],
                         fill_mode="nearest",
                         cval=0.0)

# convert the labels from ints to vectors
# encode labels
labels = np.array(labels)
le = LabelEncoder()
le.fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = float(classTotals.max()) / classTotals
print("class weight balance: {}".format(classWeight))
print("class totals: {}".format(classTotals))
print("Total Samples = {} ".format(data.shape[0]))


# partition the data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Init optimzer and model
# opt = Adam(lr=1e-4, decay=1e-4/(epochs))
opt = RMSprop(lr=1e-3, decay=1e-3/(epochs))
# model = Alexnet.build(width=227, height=227, depth=3, classes=2, reg=0.0)
model = ShallowNet2.build(width=148, height=148, depth=3, classes=2, reg=0.0)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print(model.summary())

# model_path = os.path.sep.join([output_path, "weights-{epoch:03d}-{val_acc:.4f}.hdf5"])
model_path = os.path.sep.join([OUTPUT_PATH, "bestmodel.hdf5"])
# monitor optios are val_loss, val_acc, train_loss, train_acc, etc. mode="max/min"
checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
figPath = os.path.sep.join([OUTPUT_PATH, "{}_".format(os.getpid())])
jsonPath = os.path.sep.join([OUTPUT_PATH, "{}.json".format(os.getpid())])
callbacks = [checkpoint, TrainingMonitor(figPath, jsonPath=jsonPath)]

# train
print("[INFO] training ...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // 64,
                        epochs=epochs,
                        callbacks=callbacks,
                        # class_weight=classWeight,
                        verbose=1)

# evaulation
# Load best model from training and make predictions on it
model = load_model(model_path)
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

plt.style.use("ggplot")
acc = H.history['accuracy']
val_acc = H.history['val_accuracy']
loss = H.history['loss']
val_loss = H.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Training and Validation and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.sep.join([OUTPUT_PATH, "Acc.png"]))
plt.close()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and Validation and Loss")
plt.xlabel("Epoch #")
plt.ylabel("loss")
plt.legend()
plt.savefig(os.path.sep.join([OUTPUT_PATH, "Loss.png"]))
plt.close()
