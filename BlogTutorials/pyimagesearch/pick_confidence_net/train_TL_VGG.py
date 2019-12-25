from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from BlogTutorials.pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.aspectawarepreprocessing import AspectAwarePreprocessor
from BlogTutorials.pyimagesearch.preprocessing.ROIpreprocessor import ROIPreprocessor
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
# from BlogTutorials.pyimagesearch.nn.conv.alexnet import Alexnet
from keras.applications import VGG16
from keras import models, layers
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2 # L2 normalization weight decay to cost function

"""
    Alexnet net trained on depth maps for mushrooms/end tool in frame. Attempts binary classification of a depth map,
    with output softmax to classify a prediction of the pick will be successful.

    Outline of training approach (dogs_vs_cats classifier as template)

    320 x 240 input pixels
    (add mask layer of target mushroom)
    labels org: /pick_confidence_net/pick_fail_depth & /pick_success_depth

    /config/pick_confidence_config.py
        file paths
        num classes, training/val/test ratio of data
        HDF5 data locations
        outputs: model, normalization distance values, charts/training data

    /build_pick_conf_dataset.py
        # Get image paths
        # Sep training, test, and validation data
        # datasets list
        # Preprocessors images (crop to ROI (160:320,0:120), normalize distance points)
        
"""


def normalize_depth(data):
    # normalize data [0, 1]
    return  (data - np.min(data)) / (np.ptp(data))

def normalize_rgb(data):
    return data.astype("float") / 255.0
#

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(units=256,
                kernel_regularizer=l2(0.002)))
model.add(layers.Activation("relu"))
model.add(BatchNormalization())
model.add(layers.Dropout(0.5))

# 5. FC => RELU
# model.add(layers.Dense(units=512,
#                 kernel_regularizer=l2(0.002)))
# model.add(layers.Activation("relu"))
# model.add(BatchNormalization())
# model.add(layers.Dropout(0.5))

# 6. softmax classifer
model.add(layers.Dense(2, kernel_regularizer=l2(0.002)))
model.add(layers.Activation("softmax"))
# Freeze base for initial training
conv_base.trainable = False

epochs = 100
# dataset_path = r"H:\Datasets\Pluckt\pick_confidence_net\data\depth"
dataset_path = r"H:\Datasets\Pluckt\pick_confidence_net\data\rgb"
output_path = r"H:\Datasets\Pluckt\pick_confidence_net\models"

# checkpoint_model_path = r"H:\Datasets\Pluckt\pick_confidence_net\models\vggnet\1001_examples\1\bestmodel.hdf5"
# get class labels
imagePaths = list(paths.list_images(dataset_path))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]


# init preprocessors
roip = ROIPreprocessor(xmin=0, xmax=240, ymin=80, ymax=320)
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixels
sdl = SimpleDatasetLoader(preprocessors=[roip, aap, iap]) #, img_load="mat")
(data, labels) = sdl.load(imagePaths=imagePaths, verbose=250)
data = normalize_rgb(data)
# data = normalize_depth(data)

# add data augmentation with image generator from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=0.0, # randomly rotate +/- 30 deg
                         width_shift_range=0.05, # translate by 0.1
                         height_shift_range=0.05,
                         shear_range=0.0, # shear by 0.2
                         zoom_range=0.0, # zoom by [0.8 - 1.2] factory
                         horizontal_flip=False,
                         # brightness_range=[0.9, 1.1],
                         fill_mode="nearest",
                         cval=0.0)

# encode labels
labels = np.array(labels)
le = LabelEncoder()
le.fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# accouunt for the skew in data labels. Used to amplifiy training weights of "smiling" case given the ratio of
classTotals = labels.sum(axis=0)
classWeight = float(classTotals.max()) / classTotals

# split data (stratify sampling samples at same ratio of data set for test to train split. i.e the 9475:3690 ratio
#  is kept in both the test and training set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.2, stratify=labels, random_state=42)



# Optimizer
print("[info] compiling model...")
opt = RMSprop(lr=2e-5)
# model = Alexnet.build(width=227,
#                       height=227,
#                       depth=1,
#                       classes=2,
#                       reg=0.0002)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])
# print("loading model {}...".format(checkpoint_model_path))
# model = load_model(checkpoint_model_path)

# model_path = os.path.sep.join([output_path, "weights-{epoch:03d}-{val_acc:.4f}.hdf5"])
model_path = os.path.sep.join([output_path, "bestmodel.hdf5"])
# monitor optios are val_loss, val_acc, train_loss, train_acc, etc. mode="max/min"
checkpoint = ModelCheckpoint(model_path, monitor="val_acc", mode="max", save_best_only=True, verbose=1)
figPath = os.path.sep.join([output_path, "{}_".format(os.getpid())])
jsonPath = os.path.sep.join([output_path, "{}.json".format(os.getpid())])
callbacks = [checkpoint, TrainingMonitor(figPath, jsonPath=jsonPath)]


# train
# H = model.fit(trainX,
#               trainY,
#               validation_data=(testX, testY),
#               batch_size=32,
#               epochs=epochs,
#               callbacks=callbacks,
#               verbose=1)

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=16),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // 16,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=1)

# Load best model from training and make predictions on it
model = load_model(model_path)
predictions = model.predict(testX, batch_size=16)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

plt.style.use("ggplot")
acc = H.history['acc']
val_acc = H.history['val_acc']
loss = H.history['loss']
val_loss = H.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Training and Validation and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.sep.join([output_path, "Acc.png"]))
plt.close()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and Validation and Loss")
plt.xlabel("Epoch #")
plt.ylabel("loss")
plt.legend()
plt.savefig(os.path.sep.join([output_path, "Loss.png"]))
plt.close()



