from keras.applications import VGG16
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from BlogTutorials.pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader

import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = r'H:\Datasets\kaggle_dogs_vs_cats'
train_dir = os.path.join(base_dir,'training')
train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')

validation_dir = os.path.join(base_dir, 'validation')
validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dogs_dir = os.path.join(validation_dir,'dogs')

output_path = os.path.join(base_dir, 'models')

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

bs = 20
epochs = 30
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=bs,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=bs,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

H = model.fit_generator(
    train_generator,
    steps_per_epoch=len(os.listdir(train_cats_dir))//bs,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(os.listdir(validation_cats_dir))//bs)


plt.style.use("ggplot")
acc = H.history['acc']
val_acc = H.history['val_acc']
loss = H.history['loss']
val_loss = H.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Training and Validation and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.sep.join([output_path, "Acc.png"]))
plt.close()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and Validation and Loss")
plt.xlabel("Epoch #")
plt.ylabel("loss")
plt.legend()
plt.savefig(os.path.sep.join([output_path, "Loss.png"]))
plt.close()
