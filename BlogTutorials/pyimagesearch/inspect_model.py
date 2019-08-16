from keras.applications import VGG16

include_top = 1

# load vgg16
print("[info] loading model..")
model = VGG16(weights="imagenet",
              include_top=True)
print("[info] showing layers...")

# loop over layers and display
for (i, layer) in enumerate(model.layers):
    print("{}\t{}".format(i, layer.__class__.__name__))



