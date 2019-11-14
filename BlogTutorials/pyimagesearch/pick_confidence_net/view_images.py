import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from imutils import paths
import matplotlib.animation as animation

dataset_path = r"D:\matth\Documents\projects\python\DL_Projects\Pluckt\pick_confidence_net\data"
imagePaths = list(paths.list_images(dataset_path))


def update(path):
    p = path
    displayimage = mpimg.imread(p)
    plt.gca().clear()
    plt.imshow(displayimage)


for image in imagePaths:
    # ani = animation.FuncAnimation(plt.gcf(), update, image, interval = 1000, repeat=False)
    update(image)
    plt.show()

