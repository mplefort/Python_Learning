from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.aspectawarepreprocessing import AspectAwarePreprocessor
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from BlogTutorials.pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

dataset_path = r"H:\Datasets\17flowers"

# get class labels
imagePaths = list(paths.list_images(dataset_path))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]


# init preprocessors
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixels
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths=imagePaths[1:10], verbose=250)
data = data.astype("float") / 255.0

# partision data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels to vectros
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize optimizer
print("[info] compile model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
epochs = 100
print("[info] Training model")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=epochs, verbose=1)

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

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