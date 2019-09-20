import matplotlib
matplotlib.use("Agg")
from keras.utils import plot_model
from BlogTutorials.pyimagesearch.tiny_imagenet_minigooglenet.config import tiny_imagenet_config as config
from BlogTutorials.pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from BlogTutorials.pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from BlogTutorials.pyimagesearch.preprocessing.meanpreprocessor import MeanPreprocessor
from BlogTutorials.pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from BlogTutorials.pyimagesearch.callbacks.epochcheckpoint import EpochCheckpoint
from BlogTutorials.pyimagesearch.io_module.hdf5datasetgenerator import HD5FDatasetGenerator
from BlogTutorials.pyimagesearch.nn.conv.mini2_googlenet import Mini2GoogLeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import json
from keras.callbacks import ModelCheckpoint
import os

checkpoints_path = r"D:\matth\Documents\projects\python\datasets\tiny_imagenet\outputs\checkpoints"
model_checkpoint = None #
# model_checkpoint = r"D:\matth\Documents\projects\python\datasets\tiny_imagenet\outputs\checkpoints\epoch_5.hdf5"
start_epoch = 0
epochs_to_run = 20
learning_rate = 1e-3
batch_size = 96
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

# preprocessors
sp = SimplePreprocessor(64,64)  # resize image to size
mp = MeanPreprocessor(means["R"], means["B"], means["G"])
iap = ImageToArrayPreprocessor()

# init datasets
trainGen = HD5FDatasetGenerator(config.TRAIN_HDF5, batchSize=batch_size, aug=aug,
                                preprocessor=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HD5FDatasetGenerator(config.VAL_HDF5, batchSize=batch_size, preprocessor=[sp, mp, iap],
                              classes=config.NUM_CLASSES)


if model_checkpoint is None:
    print("[info] Compiling model...")
    model = Mini2GoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES,
                                 reg=0.0002)
    opt = Adam(learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
else:
    print("loading model {}...".format(model_checkpoint))
    model = load_model(model_checkpoint)
    print("[info] old learning rate {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, learning_rate)
    print("[info] new learning rate {}".format(K.get_value(model.optimizer.lr)))


callbacks = [EpochCheckpoint(checkpoints_path, every=5, startAt=start_epoch),
             TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PAT, startAt=start_epoch)]

# train model
# plot_model(model, to_file='model.png')

model.fit_generator(trainGen.generator(),
                    steps_per_epoch=trainGen.numImages // batch_size,
                    validation_data=valGen.generator(),
                    validation_steps=valGen.numImages // batch_size,
                    epochs=epochs_to_run,
                    max_queue_size=batch_size * 2,
                    callbacks=callbacks,
                    verbose=1)

trainGen.close()
valGen.close()

