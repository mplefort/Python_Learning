from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parsers
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="Path to pretrained model")
args = vars(ap.parse_args())

# init class labels
classLabels = ["cat", "dog", "panda"]

# grab list of images
print("[info] Sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

# image preprocessors
sp = SimplePreprocessor(32, 32)  # resize image to a 32,32
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel to [0,1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths=imagePaths)
data = data.astype("float") / 255.0

# Load pretrained model
print("[INFO} loading model...")
model = load_model(args["model"])

# predictions
print("Making predictions...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
    # load the example image and put  predictionon it
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
