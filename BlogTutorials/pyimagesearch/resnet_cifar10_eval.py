from BlogTutorials.pyimagesearch.utils.ranked import rank5_accuracy
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import pickle
import h5py
from keras.datasets import cifar10
from keras.models import Model  # ilo of sequential where output of 1 is input to next
from keras.models import load_model
import numpy as np

# load CIFAR-10 Data
print("Loading CIFAR 10")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtration
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# convert labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model_path = r'D:\matth\Documents\projects\python\models\resnet_cifar10\checkpoints\epoch_55.hdf5'

print('[info] loading model')
# model = pickle.loads(open(model_path, "rb").read())
model = load_model(filepath=model_path)

print("[INFO] evaluating mdoel ...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=labelNames))
