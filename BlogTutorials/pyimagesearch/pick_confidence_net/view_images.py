import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imutils import paths
from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.aspectawarepreprocessing import AspectAwarePreprocessor
from BlogTutorials.pyimagesearch.preprocessing.ROIpreprocessor import ROIPreprocessor
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
import matplotlib.animation as animation
import numpy as np
import os

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

dataset_path = r"H:\Datasets\Pluckt\pick_confidence_net\data\rgb"
# dataset_path = r"H:\Datasets\Pluckt\pick_confidence_net\data\depth"
imagePaths = list(paths.list_images(dataset_path))
roip = ROIPreprocessor(xmin=0, xmax=240, ymin=80, ymax=320)
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixels
sdl = SimpleDatasetLoader(preprocessors=[roip, aap, iap]) #, img_load="mat")
(data, labels) = sdl.load(imagePaths=imagePaths, verbose=250)
data = normalize_rgb(data)

# imagePaths = list(paths.list_images(dataset_path))
# classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
# classNames = [str(x) for x in np.unique(classNames)]
labels = np.array(labels)


for image, label in zip(data[0:5], labels[0:5]):
    # ani = animation.FuncAnimation(plt.gcf(), update, image, interval = 1000, repeat=False)
    plt.imshow(image)
    plt.title(label)
    plt.show()

for image, label in zip(data[-5:], labels[-5:]):
    # ani = animation.FuncAnimation(plt.gcf(), update, image, interval = 1000, repeat=False)
    plt.imshow(image)
    plt.title(label)
    plt.show()

