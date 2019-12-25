from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report
from imutils import paths
from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.aspectawarepreprocessing import AspectAwarePreprocessor
from BlogTutorials.pyimagesearch.preprocessing.ROIpreprocessor import ROIPreprocessor
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.animation as animation
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

def update(path):
    p = path
    displayimage = mpimg.imread(p)
    plt.gca().clear()
    plt.imshow(displayimage)


def normalize_depth(data):
    # normalize data [0, 1]
    return  (data - np.min(data)) / (np.ptp(data))

def normalize_rgb(data):
    return data.astype("float") / 255.0

model_path = r'H:\Datasets\Pluckt\pick_confidence_net\models\vggnet\602_examples\TF_learning\1\bestmodel.hdf5'
output_path = r"H:\Datasets\Pluckt\pick_confidence_net\models\vggnet\602_examples\TF_learning\1"


dataset_path = r"H:\Datasets\Pluckt\pick_confidence_net\data\rgb"
# dataset_path = r"H:\Datasets\Pluckt\pick_confidence_net\data\depth"
imagePaths = list(paths.list_images(dataset_path))
# init preprocessors
roip = ROIPreprocessor(xmin=0, xmax=120, ymin=160, ymax=320)
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixels
sdl = SimpleDatasetLoader(preprocessors=[aap, iap]) #, img_load="mat")
(data, labels) = sdl.load(imagePaths=imagePaths, verbose=250)
data = normalize_rgb(data)
# data = normalize_depth(data)

imagePaths = list(paths.list_images(dataset_path))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]
labels = np.array(labels)
# encode labels
labels = np.array(labels)
le = LabelEncoder()
le.fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=.2, stratify=labels, random_state=42)


model = load_model(model_path)

predictions = model.predict(testX, batch_size=16)


print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

print("need to print pics")

i = 0
for image, gnd_t, pred in zip(testX, testY.argmax(axis=1), predictions):
    plt.imshow(image)
    if gnd_t == pred.argmax():
        pred_eval = "Correct"
    else:
        pred_eval = "Incorrect"
    plt.title("{} \n Gnd_t: {} \n Prediction[{}]: {} %".format(
        pred_eval, classNames[gnd_t], classNames[pred.argmax()], pred))
    plt.savefig(os.path.sep.join([output_path, "prediction_{}.png".format(i)]))
    i += 1
