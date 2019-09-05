import matplotlib
matplotlib.use("Agg")

from BlogTutorials.pyimagesearch.dogs_vs_cats.config import dogs_vs_cats_config as config

from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from BlogTutorials.pyimagesearch.preprocessing.patchpreprocessor import PatchPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor

from BlogTutorials.pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from BlogTutorials.pyimagesearch.io_module.hdf5datasetgenerator import HD5FDatasetGenerator
from BlogTutorials.pyimagesearch.nn.conv.alexnet import Alexnet

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import json
import os

# data augmentation training
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# load RGB mean
means = json.loads(open(config.DATA_SET_MEAN).read())

# preprocessors
sp = SimplePreprocessor(227, 227) # used only on validation data
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# training and validation dataset generatros
trainGen = HD5FDatasetGenerator(config.TRAIN_HDF5, 32, aug=None,
                                preprocessor=[pp, mp, iap], classes=2)
valGen = HD5FDatasetGenerator(config.VAL_HDF5, 32,
                              preprocessor=[sp, mp, iap], classes=2)

# Optimizer
print("[info] compiling model...")
opt = Adam(lr=1e-3)
model = Alexnet.build(width=227,
                      height = 227,
                      depth=3,
                      classes=2,
                      reg=0.0002)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

# callabacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
# To save model at each epoch if using ctrl+C method
# monitor optios are val_loss, val_acc, train_loss, train_acc, etc. mode="max/min"
# checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
# callbacks = [checkpoint]
callbacks = [TrainingMonitor(path)]

# train
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 32,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 32,
    epochs=40,
    max_queue_size=32*2,
    callbacks=callbacks,
    verbose=1)

# save model
print("Saving model...")
model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()