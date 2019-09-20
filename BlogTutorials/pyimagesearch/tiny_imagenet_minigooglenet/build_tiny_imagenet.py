from BlogTutorials.pyimagesearch.tiny_imagenet_minigooglenet.config import tiny_imagenet_config as config

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from BlogTutorials.pyimagesearch.io_module.hdf5datasetwriter import HDF5DatasetWriter

from imutils import paths
import numpy as np
import json
import cv2
import os

trainPaths = list(paths.list_images(config.TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-3] for p in trainPaths]
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# split training set for a testing set
split = train_test_split(trainPaths, trainLabels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=trainLabels,
                         random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# Load the validation filename class from file and then use these mappings
# to build the validation paths and label lists
M = open(config.VAL_MAPPINGS).read().strip().split("\n")
M = [r.split("\t") for r in M]
valPaths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
valLabels = le.transform([m[1] for m in M])

# contruct a list to hold ["dataset", imagepaths, labels, HDF5 path
datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)
]

# Average RGB
(R, G, B) = ([], [], [])

# loop over data tupes
for (dtype, paths, labels, outputPath) in datasets:
    print("[info] building {} dataset...".format(outputPath))
    writer = HDF5DatasetWriter(dims=(len(paths), 64, 64, 3),
                               outputPath=outputPath)

    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)

        # if training set build RGB averages
        if dtype == "train":
            R.append(cv2.mean(image)[:3][2])
            G.append(cv2.mean(image)[:3][1])
            B.append(cv2.mean(image)[:3][0])

        writer.add([image], [label])
        if i % int(len(paths)/100) == 0:
            print("[info] saved {} / {} : {} / {}%".format(i, len(paths), i/len(paths), 100))

    writer.close()

# save RGB means
print("[info] saving means RGB...")
D = {"R": np.mean(R),
     "G": np.mean(G),
     "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()



