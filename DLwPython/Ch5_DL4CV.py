from keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image
from keras import models

"""
5.4 Visualizing Convents
"""

# img_fn = "/home/matthewlefort/Pictures/Pic.jpg"
# img = cv2.imread(img_fn)
# img = cv2.resize(img, (224, 224))
#
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255
#
# model = VGG16(weights="imagenet")
#
#
#
# layer_names = []
# for layer in model.layers[1:len(model.layers):3]:
#     layer_names.append(layer.name)
#
# images_per_row = 16
# layer_outputs = [layer.output for layer in model.layers[:8]]
# activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(img_tensor)
#
# for layer_name, layer_activation in zip(layer_names, activations):
#     n_features = layer_activation.shape[-1]
#     size = layer_activation.shape[1]
#
#     n_cols = n_features // images_per_row
#     display_grid = np.zeros((size * n_cols, images_per_row*size))
#
#     for col in range(n_cols):
#         for row in range(images_per_row):
#             channel_image = layer_activation[0, :, :, col * images_per_row + row]
#             channel_image -= channel_image.mean()
#             if np.abs(channel_image.std()) > 1e-3:
#                 channel_image /= channel_image.std()
#             else:
#                 channel_image /= 1e-3
#             channel_image *= 64
#             channel_image += 128
#             channel_image = np.clip(channel_image, 0, 255).astype('uint8')
#             display_grid[col * size : (col + 1) * size,
#                        row * size : (row + 1) * size] = channel_image
#
#     scale = 1.0 / size
#     plt.figure(figsize=(scale * display_grid.shape[1],
#                         scale * display_grid.shape[0]))
#     plt.title(layer_name)
#     plt.grid(False)
#     plt.imshow(display_grid, aspect='auto', cmap='viridis')
#
# print("final ")


"""
5.4.3 Visualize by heatmap - This is cool to see input image light up 
"""
from keras.applications import VGG16
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import models
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras import backend as K

img_fn = "D:\matth\Pictures\Portfolio\Headshot.jpg"
img = cv2.imread(img_fn)
img = cv2.resize(img, (224, 224))
model = VGG16(weights="imagenet")

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)

tie_output = model.output[:, 906]
last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(tie_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

cv2.imwrite('/home/matthewlefort/Pictures/Pic.jpg', superimposed_img)