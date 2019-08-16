from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from time import sleep
from BlogTutorials.pyimagesearch.io_module.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
args = vars(ap.parse_args())
batch_size = 32
buffer_size = 1000

# List of images to extract features on
print("[info] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
# shuffle image paths because sklearn.preprocessing.test_train_split() only works with all samples in memory.
random.shuffle(imagePaths)

# extract the class labels from the image paths then encode thel labels
labels = []
for p in imagePaths:
    label = p.split(os.path.sep)[-2]
    labels.append(label)
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the network
print("[info] loading model ...")
# weights loads prese
model = VGG16(weights="imagenet", # loads imagenet pretrained network
              include_top=False)  # does not include 3 FC layers at end of standard VGG, outputs 512 x 7 x 7

# initializethe HDF5 dataset writer, then store the calss label names in the datset
dataset = HDF5DatasetWriter( (len(imagePaths), 512*7*7), #
                             args["output"],
                             dataKey="features",
                             bufSize=buffer_size)
dataset.storeClassLabels(le.classes_)

# init progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over images in patches to begin extraction
for i in np.arange(0, len(imagePaths), batch_size):
    # extract the batch of images and labels, then initialize the list of actual images that will
    # be passed through the network for feature extraction
    batchPaths = imagePaths[i:i+batch_size]
    batchLabels = labels[i:i+batch_size]
    batchImages = []

    # loop over images and labels in the current patch
    for (j, imagePath) in enumerate(batchPaths):
        # load input image using Keras helper utility while ensuring
        # image is resized to 224 x 224 pixels.
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess image by epanding dims and subtracting mean RGB pixel values (imagenet avgs)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add image to batch
        batchImages.append(image)

    # pass images through hte network and use outputs as features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=batch_size)

    # reshape feature
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    dataset.add(features, batchLabels)
    print("Progress {}".format(i/len(imagePaths)))
    sleep(0.1)

dataset.close()
pbar.finish()
