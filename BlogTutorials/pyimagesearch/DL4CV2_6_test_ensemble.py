from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import load_model
from keras.datasets import cifar10

import numpy as np
import glob
import os

models_dir = r'D:\matth\Documents\projects\python\models\miniVGG_ensemble_cifar10'

(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# model list
model_paths = os.path.sep.join([models_dir, "*.model"])
model_paths = list(glob.glob(model_paths)) # glob util to find all paths with ending in .model
models = []
# append to list of models
for (i, model_path) in enumerate(model_paths):
    models.append(load_model(model_path))
    print("[info] loaded model {} / {}".format(i, len(model_paths)))

print("[info] evaluating ensemble")
predictions = []

for model in models:
    predictions.append(model.predict(testX, batch_size=64))

predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labelNames))


