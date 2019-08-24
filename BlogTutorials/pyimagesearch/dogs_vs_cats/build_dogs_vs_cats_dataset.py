from BlogTutorials.pyimagesearch.dogs_vs_cats.config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from BlogTutorials.pyimagesearch.preprocessing.aspectawarepreprocessing import AspectAwarePreprocessor
from BlogTutorials.pyimagesearch.io_module.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths

import numpy as np
import json
import cv2
import os

# Get image paths
trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = []
for p in trainPaths:
    trainLabels.append(p.split(os.path.sep)[-1].split(".")[0])

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# Sep training, test, and validation data
(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(
    trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES,
    stratify=trainLabels, random_state=42)

(trainPaths, valPaths, trainLabels, valLabels) = train_test_split(
    trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES,
    stratify=trainLabels, random_state=42)

# datasets list
datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5) ]

# Preprocessors iamges
aap = AspectAwarePreprocessor(256, 256) # resize to 256, 256 keeping aspect (cropped)
(R,G,B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[info] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath=outputPath)

    for(i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            B.append(b)
            G.append(g)

        writer.add([image], [label])
        if i % 100 == 0:
            print("[info] saved {} / {}".format(i, len(paths)))

    writer.close()

print("[info] saving RGB means..")
D = {"R":np.mean(R), "G":np.mean(G), "B":np.mean(B)}
f = open(config.DATA_SET_MEAN,"w")
f.write(json.dumps(D))
f.close()




