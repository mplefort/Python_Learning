from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from BlogTutorials.pyimagesearch.io_module.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import random
import os

dataset_path = r'D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\train\train'
extract_feature_output = r'D:\matth\Documents\projects\python\datasets\kaggle_dogs_vs_cats\models\resnet50 _feature_extraction\catsvdogresnet.hdf5'
bs = 16 # batch size
buffer_size = 1000 # feature extraction buffer

# load images
print('[info] loading images...')
imagePaths = list(paths.list_images(dataset_path))
random.shuffle(imagePaths)

# extract labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load model
print("[info] loading resnet network...")
model = ResNet50(weights="imagenet",
                 include_top=False,
                 pooling="max")

# HDF5 writer to store output features
dataset = HDF5DatasetWriter(dims=(len(imagePaths), 2048),   #output of resnet50 is 2048
                            outputPath=extract_feature_output,
                            dataKey="features",
                            bufSize=bs,)
dataset.storeClassLabels(le.classes_)

# extract features
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]
    batchImages = []

    # preprocess images in batch before passing to model
    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add image to batch
        batchImages.append(image)

    # pass images to resnet
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    # flatten maxpool output to 1,2048
    features = features.reshape((features.shape[0], 2048))

    dataset.add(features, batchLabels)
    if i % bs*4 == 0:
        print("Progress: {}/{}".format(i, len(imagePaths)))

dataset.close()


