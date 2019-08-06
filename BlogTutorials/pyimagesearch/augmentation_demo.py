from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import numpy as np

image_path = r"D:\matth\Pictures\Phone Photos\IMG_1203.JPG"
output_path = r"D:\matth\Documents\projects\python\models\image_augmentation_demo"
output_filename = "output"

print("[info] load image...")
image = load_img(image_path)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# construct image generatro for data aug
aug = ImageDataGenerator( rotation_range=30,      # degrees random rotation
                          width_shift_range=0.1,  # horizontal shifts
                          height_shift_range=0.1,  # vertical shifts
                          shear_range=0.2,        # angle CCW dir radians to shear
                          zoom_range=0.2,         # zoom range is 1 +/- zoom_range
                          horizontal_flip=True,
                          fill_mode="nearest")

print("[info] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=output_path,
                    save_prefix=output_filename, save_format="jpg")
total = 0
for image in imageGen:
    total += 1
    if total == 10:
        break













